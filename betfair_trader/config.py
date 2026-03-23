"""Configuration management using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BetfairConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Betfair credentials
    username: str = Field(alias="BETFAIR_USERNAME")
    password: str = Field(alias="BETFAIR_PASSWORD")
    app_key: str = Field(alias="BETFAIR_APP_KEY")
    certs_path: str = Field(default="./certs", alias="BETFAIR_CERTS_PATH")

    # Anthropic
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Trading
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")
    max_stake: float = Field(default=2.0, alias="MAX_STAKE")
    max_liability: float = Field(default=10.0, alias="MAX_LIABILITY")
    kelly_fraction: float = Field(default=0.25, alias="KELLY_FRACTION")
    commission_rate: float = Field(default=0.05, alias="BETFAIR_COMMISSION")

    # Strategy
    min_edge: float = Field(default=0.05, alias="MIN_EDGE")
    min_confidence: float = Field(default=0.6, alias="MIN_CONFIDENCE")
    scalp_ticks: int = Field(default=2, alias="SCALP_TICKS")
