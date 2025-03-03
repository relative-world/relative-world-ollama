from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    base_url: str = "http://localhost:11434"
    default_model: str = "qwen2.5:14b"  # we do what we can
    json_fix_model: str = "qwen2.5:14b"
    model_keep_alive: float = 300.0

    model_config = SettingsConfigDict()
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_prefix = "relative_world_ollama_"
    )

settings = OllamaSettings()
