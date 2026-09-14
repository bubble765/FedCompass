"""Natural-language experiment configuration assistant API."""

from fastapi import APIRouter, HTTPException

from app.agent.config_assistant import ConfigAssistantError, parse_config_request
from app.schemas.schemas import ConfigAssistantRequest, ConfigAssistantResponse

router = APIRouter(prefix="/api", tags=["config-assistant"])


@router.post("/experiment-config-assistant/parse", response_model=ConfigAssistantResponse)
async def parse_experiment_config(request: ConfigAssistantRequest):
    try:
        return await parse_config_request(request)
    except ConfigAssistantError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

