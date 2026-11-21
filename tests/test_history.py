"""Scheduling history: persistence, dedupe, and transactional integrity."""

from __future__ import annotations

import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app.db import Store
from app.main import create_app
from app.scheduler import JobNotFoundError, Scheduler
from tests.conftest import FakeClock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def events(scheduler, job_id):
    return scheduler.job_history(job_id)["events"]


def event_types(scheduler, job_id):
    return [e["event_type"] for e in events(scheduler, job_id)]


def waiting_rows(scheduler, job_id):
    return [
        (e["queue_position"], e["blocked_reason"])
        for e in events(scheduler, job_id)
        if e["event_type"] == "waiting"
    ]


def raw_history(scheduler, job_id):
    """History straight from SQLite, without triggering a scheduling pass."""
    conn = scheduler.store._connect()
    try:
        rows = conn.execute(
            "SELECT event_type FROM job_events WHERE job_id = ? ORDER BY seq",
            (job_id,),
        ).fetchall()
        return [row["event_type"] for row in rows]
    finally:
        conn.close()


def raw_job_status(scheduler, job_id):
    conn = scheduler.store._connect()
    try:
        row = conn.execute(
            "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return row["status"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# What gets recorded
# ---------------------------------------------------------------------------


def test_immediate_dispatch_records_queued_and_dispatched(scheduler):
    scheduler.configure_node("n1", 4)
    scheduler.submit_job("a", 2, 30)

    a_events = events(scheduler, "a")
    assert [e["event_type"] for e in a_events] == ["queued", "dispatched"]
    queued, dispatched = a_events
    assert queued["status"] == "queued"
    assert queued["queue_position"] == 1
    assert queued["occurred_at"] == scheduler.get_job("a")["submitted_at"]
    assert dispatched["status"] == "running"
    assert dispatched["node_id"] == "n1"
    assert dispatched["slots"] == 2
    job = scheduler.get_job("a")
    assert dispatched["occurred_at"] == job["started_at"]
    assert dispatched["lease_expires_at"] == job["lease_expires_at"]


def test_waiting_records_position_and_reason_until_dispatch(scheduler):
    scheduler.configure_node("n1", 2)
    scheduler.submit_job("a", 2, 60)
    scheduler.submit_job("b", 2, 60)

    b_events = events(scheduler, "b")
    assert [e["event_type"] for e in b_events] == ["queued", "waiting"]
    waiting = b_events[1]
    assert waiting["status"] == "queued"
    assert waiting["queue_position"] == 1
    assert waiting["blocked_reason"] == "waiting_for_free_slots"

    scheduler.release_job("a")
    assert event_types(scheduler, "b") == ["queued", "waiting", "dispatched"]


def test_waiting_dedupes_identical_records(scheduler, clock):
    scheduler.configure_node("n1", 2)
    scheduler.submit_job("a", 2, 60)
    scheduler.submit_job("b", 2, 60)

    # Repeated passes with an unchanged situation must not append rows.
    for _ in range(5):
        scheduler.tick()
        clock.advance(1)
    # Repeated queries run scheduling passes too; still no duplicates.
    for _ in range(3):
        scheduler.job_history("b")
        scheduler.get_job("b")
        scheduler.queue_state()

    assert waiting_rows(scheduler, "b") == [(1, "waiting_for_free_slots")]


def test_waiting_records_position_and_reason_changes(scheduler):
    scheduler.configure_node("n1", 2)
    scheduler.submit_job("a", 2, 60)
    scheduler.submit_job("b", 2, 60)
    scheduler.submit_job("c", 2, 60)

    # c sits behind the blocked head b.
    assert waiting_rows(scheduler, "c") == [(2, "queued_behind_head")]

    scheduler.release_job("a")  # b starts; c becomes the head
    assert waiting_rows(scheduler, "c") == [
        (2, "queued_behind_head"),
        (1, "waiting_for_free_slots"),
    ]


def test_reason_transitions_as_nodes_appear(scheduler):
    scheduler.submit_job("j", 4, 10)  # no nodes configured at all
    assert waiting_rows(scheduler, "j") == [(1, "no_nodes_configured")]

    scheduler.configure_node("n1", 2)  # too small for j
    assert waiting_rows(scheduler, "j") == [
        (1, "no_nodes_configured"),
        (1, "insufficient_capacity"),
    ]

    scheduler.configure_node("n1", 4)  # now it fits
    assert event_types(scheduler, "j") == [
        "queued",
        "waiting",
        "waiting",
        "dispatched",
    ]


def test_runaround_head_still_reports_waiting_for_slots(scheduler):
    scheduler.configure_node("n1", 4)
    scheduler.submit_job("a", 3, 60)  # running, one slot free
    scheduler.submit_job("giant", 5, 60)  # can never fit on n1
    scheduler.submit_job("small", 2, 60)  # fits a node, but no slots free now

    assert waiting_rows(scheduler, "giant") == [(1, "insufficient_capacity")]
    # The head protects no finite start time, so a still-queued later job
    # is simply short of free slots itself.
    assert waiting_rows(scheduler, "small") == [(2, "waiting_for_free_slots")]


def test_release_records_released_event(scheduler):
    scheduler.configure_node("n1", 2)
    scheduler.submit_job("a", 2, 60)
    scheduler.release_job("a")

    a_events = events(scheduler, "a")
    assert [e["event_type"] for e in a_events] == [
        "queued",
        "dispatched",
        "released",
    ]
    released = a_events[-1]
    assert released["status"] == "completed"
    assert released["node_id"] == "n1"
    assert released["slots"] == 2
    assert released["occurred_at"] == scheduler.get_job("a")["ended_at"]


def test_expiry_records_expired_event(scheduler, clock):
    scheduler.configure_node("n1", 2)
    scheduler.submit_job("a", 2, 5)
    scheduler.submit_job("b", 2, 5)
    clock.advance(5)
    scheduler.tick()

    a_events = events(scheduler, "a")
    assert [e["event_type"] for e in a_events] == [
        "queued",
        "dispatched",
        "expired",
    ]
    expired = a_events[-1]
    assert expired["status"] == "expired"
    assert expired["node_id"] == "n1"
    a = scheduler.get_job("a")
    assert expired["occurred_at"] == a["ended_at"] == a["lease_expires_at"]
    # The job queued behind a waited, then was dispatched in the same pass.
    assert event_types(scheduler, "b") == ["queued", "waiting", "dispatched"]


def test_history_unknown_job_raises(scheduler):
    with pytest.raises(JobNotFoundError):
        scheduler.job_history("missing")


# ---------------------------------------------------------------------------
# Failure handling: a rolled-back pass leaves no orphaned history
# ---------------------------------------------------------------------------


def test_failed_dispatch_leaves_no_orphan_history(scheduler, clock, monkeypatch):
    scheduler.configure_node("n1", 4)
    scheduler.submit_job("a", 2, 10)
    scheduler.submit_job("b", 2, 10)  # a and b fill n1
    scheduler.submit_job("c", 2, 10)  # queued

    assert raw_history(scheduler, "a") == ["queued", "dispatched"]
    assert raw_history(scheduler, "c") == ["queued", "waiting"]
    clock.advance(10)  # a and b are now due

    def explode(*_args, **_kwargs):
        raise RuntimeError("simulated dispatch failure")

    monkeypatch.setattr(scheduler, "_allocate", explode)
    assert scheduler.tick() == []

    # Everything the failed pass wrote (expiry for a/b, dispatch for c, any
    # waiting refresh) rolled back with it.
    assert raw_history(scheduler, "a") == ["queued", "dispatched"]
    assert raw_history(scheduler, "b") == ["queued", "dispatched"]
    assert raw_history(scheduler, "c") == ["queued", "waiting"]
    assert raw_job_status(scheduler, "a") == "running"
    assert raw_job_status(scheduler, "c") == "queued"

    monkeypatch.undo()
    scheduler.tick()
    assert event_types(scheduler, "a") == ["queued", "dispatched", "expired"]
    assert event_types(scheduler, "c") == ["queued", "waiting", "dispatched"]


def test_armed_failure_blocks_dispatch_until_disarmed(scheduler):
    scheduler.configure_node("n1", 2)
    scheduler.submit_job("a", 2, 60)
    scheduler.submit_job("b", 2, 60)

    assert scheduler.arm_dispatch_failures(100) == 100
    # The release commits (its own transaction); every follow-up dispatch
    # pass fails and rolls back, so b stays queued despite the free slots.
    released = scheduler.release_job("a")
    assert released["status"] == "completed"
    assert raw_job_status(scheduler, "b") == "queued"
    assert raw_history(scheduler, "b") == ["queued", "waiting"]
    assert raw_history(scheduler, "a") == ["queued", "dispatched", "released"]

    assert scheduler.arm_dispatch_failures(0) == 0  # disarm
    scheduler.tick()
    assert scheduler.get_job("b")["status"] == "running"
    assert event_types(scheduler, "b") == ["queued", "waiting", "dispatched"]


def test_single_armed_failure_rolls_back_one_pass(scheduler, clock):
    scheduler.configure_node("n1", 2)
    scheduler.submit_job("a", 2, 10)
    scheduler.submit_job("b", 2, 10)
    clock.advance(10)  # a is due; b waits behind it

    scheduler.arm_dispatch_failures(1)
    assert scheduler.tick() == []  # the whole pass rolled back
    assert raw_job_status(scheduler, "a") == "running"
    assert raw_history(scheduler, "a") == ["queued", "dispatched"]
    assert raw_history(scheduler, "b") == ["queued", "waiting"]

    events = scheduler.tick()  # the retry succeeds
    types = {(e["type"], e["job_id"]) for e in events}
    assert ("expired", "a") in types and ("dispatched", "b") in types
    # Exactly one record per state change — the failed attempt left nothing.
    assert event_types(scheduler, "a") == ["queued", "dispatched", "expired"]
    assert event_types(scheduler, "b") == ["queued", "waiting", "dispatched"]


# ---------------------------------------------------------------------------
# Persistence across restarts
# ---------------------------------------------------------------------------


def test_history_survives_restart(db_path, clock):
    first = Scheduler(Store(db_path), clock=clock)
    first.configure_node("n1", 2)
    first.submit_job("a", 2, 60)
    first.submit_job("b", 2, 60)

    # A brand-new Store/Scheduler on the same file sees the full history.
    second = Scheduler(Store(db_path), clock=clock)
    assert event_types(second, "b") == ["queued", "waiting"]
    assert waiting_rows(second, "b") == [(1, "waiting_for_free_slots")]

    second.release_job("a")
    assert event_types(second, "b") == ["queued", "waiting", "dispatched"]
    assert event_types(second, "a") == ["queued", "dispatched", "released"]


# ---------------------------------------------------------------------------
# Concurrency: history stays consistent under parallel ticks
# ---------------------------------------------------------------------------


def test_concurrent_ticks_keep_history_consistent(db_path, clock):
    scheduler = Scheduler(Store(db_path), clock=clock)
    scheduler.configure_node("n1", 5)
    scheduler.configure_node("n2", 3)
    for i in range(20):
        scheduler.submit_job(f"job-{i:02d}", 1 + (i % 3), 5)

    errors: list[BaseException] = []

    def worker() -> None:
        local = Scheduler(Store(db_path), clock=clock)
        try:
            for _ in range(10):
                local.tick()
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    clock.advance(5.5)  # every lease expires at once
    for thread in threads:
        thread.join()

    assert not errors
    for item in scheduler.list_jobs():
        job_events = events(scheduler, item["job_id"])
        assert job_events[0]["event_type"] == "queued"
        seqs = [e["seq"] for e in job_events]
        assert seqs == sorted(seqs)
        types = [e["event_type"] for e in job_events]
        # Each state change is recorded exactly once, even with 8
        # schedulers racing on the same database file.
        assert types.count("dispatched") <= 1
        assert types.count("expired") <= 1
        assert types.count("released") <= 1
        if item["status"] == "running":
            assert types[-1] == "dispatched"
        elif item["status"] == "expired":
            assert types[-1] == "expired"
        elif item["status"] == "queued":
            assert "dispatched" not in types
        # Consecutive waiting records never repeat position+reason.
        waits = [
            (e["queue_position"], e["blocked_reason"])
            for e in job_events
            if e["event_type"] == "waiting"
        ]
        for previous, current in zip(waits, waits[1:]):
            assert previous != current


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


@pytest.fixture
def api(db_path: str, clock: FakeClock):
    app = create_app(db_path=db_path, start_background=False, clock=clock)
    with TestClient(app) as client:
        client.clock = clock  # type: ignore[attr-defined]
        yield client


def configure(api, node_id: str, capacity: int):
    response = api.post("/nodes", json={"node_id": node_id, "capacity": capacity})
    assert response.status_code == 200, response.text
    return response.json()


def submit(api, job_id: str, slots: int, duration: float):
    response = api.post(
        "/jobs",
        json={
            "job_id": job_id,
            "required_slots": slots,
            "max_duration_seconds": duration,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_history_and_fault_injection_in_openapi(api):
    schema = api.get("/openapi.json").json()
    assert "/jobs/{job_id}/history" in schema["paths"]
    assert "/admin/fault-injection" in schema["paths"]


def test_history_endpoint_end_to_end(api):
    configure(api, "farm-1", 6)
    submit(api, "plate-001", 2, 120)
    submit(api, "plate-002", 5, 30)

    # Waiting: the head cannot start until plate-001 frees its slots.
    history = api.get("/jobs/plate-002/history").json()
    assert history["job_id"] == "plate-002"
    assert [e["event_type"] for e in history["events"]] == ["queued", "waiting"]
    waiting = history["events"][1]
    assert waiting["queue_position"] == 1
    assert waiting["blocked_reason"] == "waiting_for_free_slots"

    # Backfill: a short job runs now without delaying the head.
    submit(api, "preview-001", 2, 5)
    history = api.get("/jobs/preview-001/history").json()
    assert [e["event_type"] for e in history["events"]] == [
        "queued",
        "dispatched",
    ]

    # The backfilled job expires on schedule.
    api.clock.advance(5)
    api.post("/admin/tick")
    history = api.get("/jobs/preview-001/history").json()
    assert [e["event_type"] for e in history["events"]] == [
        "queued",
        "dispatched",
        "expired",
    ]

    # Release: the blocker returns early, the head is dispatched.
    assert api.post("/jobs/plate-001/release").json()["status"] == "completed"
    history = api.get("/jobs/plate-002/history").json()
    assert [e["event_type"] for e in history["events"]] == [
        "queued",
        "waiting",
        "dispatched",
    ]
    released = api.get("/jobs/plate-001/history").json()["events"][-1]
    assert released["event_type"] == "released"
    assert released["status"] == "completed"

    # Expiry: the lease runs out and is swept.
    api.clock.advance(30)
    api.post("/admin/tick")
    history = api.get("/jobs/plate-002/history").json()
    assert [e["event_type"] for e in history["events"]] == [
        "queued",
        "waiting",
        "dispatched",
        "expired",
    ]
    expired = history["events"][-1]
    job = api.get("/jobs/plate-002").json()
    assert expired["occurred_at"] == job["ended_at"] == job["lease_expires_at"]

    assert api.get("/jobs/missing/history").status_code == 404


def test_repeated_queries_do_not_duplicate_waiting(api):
    configure(api, "n1", 2)
    submit(api, "a", 2, 60)
    submit(api, "b", 2, 60)

    first = api.get("/jobs/b/history").json()
    for _ in range(5):
        api.post("/admin/tick")
        again = api.get("/jobs/b/history").json()
    assert again == first
    waits = [e for e in again["events"] if e["event_type"] == "waiting"]
    assert len(waits) == 1


def test_fault_injection_endpoint_rolls_back_then_recovers(api):
    configure(api, "n1", 2)
    submit(api, "a", 2, 60)
    submit(api, "b", 2, 60)

    armed = api.post("/admin/fault-injection", json={"dispatch_failures": 100})
    assert armed.status_code == 200
    assert armed.json() == {"armed_dispatch_failures": 100}

    # The release commits; the follow-up dispatch of b keeps rolling back.
    assert api.post("/jobs/a/release").json()["status"] == "completed"
    assert api.get("/jobs/b").json()["status"] == "queued"
    history = api.get("/jobs/b/history").json()
    assert [e["event_type"] for e in history["events"]] == ["queued", "waiting"]

    # Disarm and recover: exactly one dispatched record appears.
    disarmed = api.post("/admin/fault-injection", json={"dispatch_failures": 0})
    assert disarmed.json() == {"armed_dispatch_failures": 0}
    api.post("/admin/tick")
    assert api.get("/jobs/b").json()["status"] == "running"
    history = api.get("/jobs/b/history").json()
    assert [e["event_type"] for e in history["events"]] == [
        "queued",
        "waiting",
        "dispatched",
    ]


def test_fault_injection_rejects_bad_counts(api):
    assert (
        api.post("/admin/fault-injection", json={"dispatch_failures": -1}).status_code
        == 422
    )
    assert api.post("/admin/fault-injection", json={}).status_code == 422


def test_sqlite_rejects_invalid_history_rows(scheduler):
    scheduler.configure_node("n1", 1)
    scheduler.submit_job("a", 1, 10)
    conn = scheduler.store._connect()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO job_events (job_id, event_type, status,"
                " occurred_at_us) VALUES ('a', 'bogus', 'queued', 1)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO job_events (job_id, event_type, status,"
                " occurred_at_us, blocked_reason)"
                " VALUES ('a', 'waiting', 'queued', 1, 'bogus')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO job_events (job_id, event_type, status,"
                " occurred_at_us) VALUES ('ghost', 'queued', 'queued', 1)"
            )
    finally:
        conn.close()
