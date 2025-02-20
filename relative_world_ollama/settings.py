from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    base_url: str = "http://192.168.1.14:11434"
    default_model: str = "qwen2.5:14b"
    json_fix_model: str = "qwen2.5:14b"
    model_keep_alive: float = 300.0

    jinja2_template_application: str = "relative_world_ollama"
    jinja2_template_dir: str = "templates"

    model_config = SettingsConfigDict(env_prefix='relative_world_ollama_')

settings = OllamaSettings()
