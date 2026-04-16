"""Application settings using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IBSettings(BaseSettings):
    """Interactive Brokers API connection settings."""

    host: str = Field(description="IB Gateway/TWS host")
    port: int = Field(description="IB Gateway/TWS port (7497 for paper, 7496 for live)")
    client_id: int = Field(description="Client ID for IB connection")

    model_config = SettingsConfigDict(
        env_prefix="IB_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


class ContractSettings(BaseSettings):
    """Contract settings for market data requests."""

    symbol: str = Field(description="Stock symbol")
    sec_type: str = Field(description="Security type (STK, OPT, etc.)")
    exchange: str = Field(description="Exchange")
    currency: str = Field(description="Currency")

    model_config = SettingsConfigDict(
        env_prefix="CONTRACT_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = Field(description="Enable debug mode")
    log_level: str = Field(description="Logging level")
    timeout: int = Field(description="Connection timeout in seconds")

    # Nested settings
    ib: IBSettings = Field(description="IB connection settings")
    contract: ContractSettings = Field(description="Contract settings")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
