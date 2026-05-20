from pydantic_settings import BaseSettings
from decimal import Decimal


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/trade_risk_monitor"
    REDIS_URL: str = "redis://localhost:6379/0"
    VAR_THRESHOLD: Decimal = Decimal("1000000.0000")
    CONCENTRATION_THRESHOLD: Decimal = Decimal("0.2500")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
