"""Raw federated-learning state snapshots for the LLM security monitor.

This module intentionally does not classify severity or recommend a runner
action.  It only records the fields that the security-monitoring skill needs
to inspect.  The classification is produced by the security-monitor LLM.
"""

from __future__ import annotations

from typing import Any


ROUND_SNAPSHOT_SCHEMA_VERSION = "fl_round_snapshot.v1"


def build_round_snapshot(
    config: dict[str, Any],
    round_num: int,
    metrics: dict[str, Any],
    events: list[tuple[str, dict]],
) -> dict[str, Any]:
    """Build a bounded semantic snapshot without deciding whether it is safe."""
    selected_config = {
        key: config.get(key)
        for key in (
            "module_id",
            "task_type",
            "dataset",
            "split",
            "algorithm",
            "defense",
            "attack",
            "num_clients",
            "participation_rate",
            "rounds",
            "local_epochs",
            "seed",
        )
        if key in config
    }
    event_records = [
        {"event_type": event_type, "payload": payload or {}}
        for event_type, payload in events[-30:]
    ]
    return {
        "schema_version": ROUND_SNAPSHOT_SCHEMA_VERSION,
        "round": round_num,
        "scope": "fl_training",
        "approved_config_fields": selected_config,
        "observed_metrics": {
            str(name): value
            for name, value in list((metrics or {}).items())[:80]
        },
        "observed_events": event_records,
        "inspection_note": "这是待安全监控智能体审查的原始运行状态，不是已经判定的安全结论。",
    }
