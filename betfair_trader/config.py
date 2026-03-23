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

    # Racing API (form data)
    racing_api_key: str = Field(default="", alias="RACING_API_KEY")

    # Race selection filters
    filter_min_liquidity: float = Field(default=50000, alias="FILTER_MIN_LIQUIDITY")
    filter_min_class: int = Field(default=1, alias="FILTER_MIN_CLASS")
    filter_max_class: int = Field(default=4, alias="FILTER_MAX_CLASS")
    filter_min_field: int = Field(default=6, alias="FILTER_MIN_FIELD")
    filter_max_field: int = Field(default=16, alias="FILTER_MAX_FIELD")
    filter_flat_only: bool = Field(default=True, alias="FILTER_FLAT_ONLY")
    filter_exclude_maidens: bool = Field(default=True, alias="FILTER_EXCLUDE_MAIDENS")

    # Steam detection thresholds
    steam_threshold: float = Field(default=0.15, alias="STEAM_THRESHOLD")
    drift_threshold: float = Field(default=-0.20, alias="DRIFT_THRESHOLD")
    volume_threshold: float = Field(default=0.10, alias="VOLUME_THRESHOLD")
