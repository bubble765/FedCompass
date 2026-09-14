from pydantic_settings import BaseSettings


from pathlib import Path


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./fedcompass.db"
    real_data_root: str = ""
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8080", "http://127.0.0.1:5500", "*"]
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "api"
    llm_local_base_url: str = "http://127.0.0.1:8001/v1"
    llm_local_model: str = "Qwen/Qwen3-0.6B"
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 20.0

    # Background workflow agents use the local Qwen service directly. The
    # experiment scenario agent intentionally keeps using the separate
    # request/API/local switch above.
    agent_llm_provider: str = "local"
    agent_llm_local_base_url: str = "http://127.0.0.1:8001/v1"
    agent_llm_local_model: str = "Qwen/Qwen3-0.6B"
    agent_llm_api_key: str = ""
    agent_llm_base_url: str = ""
    agent_llm_model: str = "Qwen/Qwen3-0.6B"
    agent_llm_max_tokens: int = 1600
    # Qwen3-0.6B runs locally on CPU in the default development setup. A
    # structured inspection snapshot can take longer than the API default
    # while the model processes the prompt and emits JSON.
    # CPU Qwen may need more time for the larger runtime security snapshot.
    # Keep this configurable so GPU deployments can lower it if desired.
    agent_llm_timeout_seconds: float = 180.0

    # LLM evidence reviews happen at the end of this many rounds. The final
    # round is always reviewed as well.
    runtime_agent_check_interval_rounds: int = 3

    class Config:
        env_file = ".env"


settings = Settings()


def get_real_data_root() -> Path:
    """Resolve the data directory from deployment config with a local fallback."""
    configured_root = settings.real_data_root.strip()
    if configured_root:
        return Path(configured_root).expanduser()
    return Path(__file__).resolve().parents[3] / "real_data"
