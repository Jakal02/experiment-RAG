from pydantic_settings import BaseSettings, SettingsConfigDict


class APICredentials(BaseSettings):
    """Class to hold API credentials."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )

    HUGGING_FACE_API_TOKEN: str

creds = APICredentials()
