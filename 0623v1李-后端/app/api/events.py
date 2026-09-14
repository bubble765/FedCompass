"""SSE Events API."""
import asyncio
import json
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.core.database import async_session
from app.services import services
from app.models.models import ExperimentStatus

router = APIRouter(prefix="/api", tags=["events"])


@router.get("/experiments/{exp_id}/events/stream")
async def event_stream(exp_id: str):
    async with async_session() as db:
        exp = await services.get_experiment(db, exp_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")

    async def generate():
        async with async_session() as db:
            existing_logs = await services.get_logs(db, exp_id, limit=1)
            last_event_id = max((log.id for log in existing_logs), default=0)

        while True:
            async with async_session() as db:
                logs = await services.get_logs(db, exp_id, limit=100)
                new_logs = sorted((l for l in logs if l.id > last_event_id), key=lambda item: item.id)
                for log in new_logs:
                    yield {
                        "event": log.event_type,
                        "data": json.dumps(log.payload_json, default=str),
                    }
                    last_event_id = max(last_event_id, log.id)

                exp = await services.get_experiment(db, exp_id)
                # `completed` is an internal runner milestone. The post-run
                # operations, security and result agents still need to emit
                # their events, so keep SSE open until the workflow reaches
                # analyzed (or an actually terminal failure/stop state).
                if exp and exp.status in (ExperimentStatus.analyzed, ExperimentStatus.failed, ExperimentStatus.stopped):
                    yield {"event": "done", "data": json.dumps({"status": exp.status.value})}
                    break

            await asyncio.sleep(1)

    return EventSourceResponse(generate())
