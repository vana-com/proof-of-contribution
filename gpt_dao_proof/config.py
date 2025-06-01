"""Application configuration and environment settings"""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    DB_PASSWORD: str = Field(..., description="Database password")

    # TEE context
    DLP_ID: int = Field(default=89, description="Data Liquidity Pool ID for GPT DAO. 89 - moksha testnet, ... - mainnet")
    FILE_ID: Optional[int] = Field(default=0, description="File ID being processed")
    JOB_ID: Optional[str] = Field(default="local-job", description="TEE job ID")
    OWNER_ADDRESS: Optional[str] = Field(default="0xUNKNOWN", description="Owner's wallet address")

    # Directories
    INPUT_DIR: str = Field("/input", description="Directory containing input files from user export")
    OUTPUT_DIR: str = Field("/output", description="Directory for PoC output results.json")
    DEBUG_DIR: str = Field("./debug_gpt_dao", description="Directory for debug files (local dev only)")

    # Scoring thresholds
    MAX_INFOGRAPHIC_POINTS: int = Field(1000, description="Denominator for normalizing raw points to a 0-1 score for the contract and infographic representation.")
    WORD_CLOUD_TOP_N: int = Field(30, description="Number of top words for the word cloud")
    BADGE_GPT4_USAGE_THRESHOLD_PCT: float = Field(0.5, description="Min % of GPT-4 messages for Power User badge")
    BADGE_FEEDBACK_BRONZE_COUNT: int = Field(10)
    BADGE_FEEDBACK_SILVER_COUNT: int = Field(50)
    BADGE_FEEDBACK_GOLD_COUNT: int = Field(100)
    BADGE_CONSISTENT_DAYS_THRESHOLD: int = Field(90)  # 3 months
    BADGE_RAPID_RESPONDER_INTERACTIONS_THRESHOLD: int = Field(10)  # >10 interactions in last 7 days

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )

settings = Settings()