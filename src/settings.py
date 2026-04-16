"""Application settings using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Main application settings."""

    # Application settings
    debug: bool = Field(description="Enable debug mode")
    log_level: str = Field(description="Logging level")
    timeout: int = Field(description="Connection timeout in seconds")

    # IB Connection settings
    ib_host: str = Field(description="IB Gateway/TWS host")
    ib_port: int = Field(
        description="IB Gateway/TWS port (7497 for paper, 7496 for live)"
    )
    ib_client_id: int = Field(description="Client ID for IB connection")

    model_config = SettingsConfigDict(
        env_file=".env.example",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


if __name__ == "__main__":
    from rich import print

    settings = AppSettings()
    print(settings)
