from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    LLAMA_BASE_URL: str = "http://127.0.0.1:8080"
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


config = Config()
