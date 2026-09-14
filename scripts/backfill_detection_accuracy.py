#!/usr/bin/env python3
"""Backfill historical detection_accuracy values for defense experiments."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = WORKSPACE_ROOT / "0623v1李-后端" / "fedcompass.db"


def parse_json_field(raw: str | None) -> list[int]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    parsed: list[int] = []
    for item in value:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return parsed


def compute_detection_accuracy(malicious: list[int], filtered: list[int]) -> float:
    malicious_set = set(malicious)
    filtered_set = set(filtered)
    if not malicious_set:
        return 0.0

    true_positive = len(malicious_set & filtered_set)
    false_positive = len(filtered_set - malicious_set)
    false_negative = len(malicious_set - filtered_set)
    denom = true_positive + false_positive + false_negative
    if denom <= 0:
        return 0.0
    return round(true_positive / denom, 4)


def main() -> int:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    experiments = cur.execute(
        """
        select id, algorithm, defense, attack
        from experiments
        where (defense is not null and defense != 'none')
           or (attack is not null and attack != 'none')
        """
    ).fetchall()

    updated_metric_rows = 0
    updated_event_rows = 0
    touched_experiments = 0

    for exp in experiments:
        exp_id = exp["id"]
        event_rows = cur.execute(
            """
            select id, event_type, payload_json
            from experiment_events
            where experiment_id = ?
              and event_type in ('client_participation', 'round_complete', 'defense_detection', 'completed')
            order by id asc
            """,
            (exp_id,),
        ).fetchall()

        malicious_by_round: dict[int, list[int]] = {}
        round_complete_events: list[sqlite3.Row] = []
        defense_detection_events: list[sqlite3.Row] = []
        completed_event: sqlite3.Row | None = None

        for row in event_rows:
            payload = json.loads(row["payload_json"] or "{}")
            event_type = row["event_type"]
            if event_type == "client_participation":
                round_num = int(payload.get("round", 0))
                malicious_by_round[round_num] = parse_json_field(json.dumps(payload.get("malicious", [])))
            elif event_type == "round_complete":
                round_complete_events.append(row)
            elif event_type == "defense_detection":
                defense_detection_events.append(row)
            elif event_type == "completed":
                completed_event = row

        if not round_complete_events:
            continue

        exp_touched = False
        final_detection = 0.0

        for row in round_complete_events:
            payload = json.loads(row["payload_json"] or "{}")
            round_num = int(payload.get("round", 0))
            filtered = parse_json_field(json.dumps(payload.get("filtered_clients", [])))
            malicious = malicious_by_round.get(round_num, [])
            detection = compute_detection_accuracy(malicious, filtered)

            cur.execute(
                """
                update experiment_metrics
                set metric_value = ?
                where experiment_id = ?
                  and round = ?
                  and metric_name = 'detection_accuracy'
                """,
                (detection, exp_id, round_num),
            )
            updated_metric_rows += cur.rowcount

            payload["detection_accuracy"] = detection
            cur.execute(
                "update experiment_events set payload_json = ? where id = ?",
                (json.dumps(payload, ensure_ascii=False), row["id"]),
            )
            updated_event_rows += cur.rowcount
            exp_touched = True
            final_detection = detection

        for row in defense_detection_events:
            payload = json.loads(row["payload_json"] or "{}")
            round_num = int(payload.get("round", 0))
            filtered = parse_json_field(json.dumps(payload.get("filtered_clients", [])))
            malicious = malicious_by_round.get(round_num, [])
            payload["detection_accuracy"] = compute_detection_accuracy(malicious, filtered)
            cur.execute(
                "update experiment_events set payload_json = ? where id = ?",
                (json.dumps(payload, ensure_ascii=False), row["id"]),
            )
            updated_event_rows += cur.rowcount
            exp_touched = True

        if completed_event is not None:
            payload = json.loads(completed_event["payload_json"] or "{}")
            payload["detection_accuracy"] = final_detection
            cur.execute(
                "update experiment_events set payload_json = ? where id = ?",
                (json.dumps(payload, ensure_ascii=False), completed_event["id"]),
            )
            updated_event_rows += cur.rowcount
            exp_touched = True

        if exp_touched:
            touched_experiments += 1

    conn.commit()
    conn.close()

    print(
        json.dumps(
            {
                "db_path": str(DB_PATH),
                "touched_experiments": touched_experiments,
                "updated_metric_rows": updated_metric_rows,
                "updated_event_rows": updated_event_rows,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
