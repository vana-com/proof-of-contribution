"""SQLAlchemy database models for storing GPT DAO contribution data."""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserGPTContribution(Base):
    """
    Tracks user ChatGPT data contributions and cumulative scores/metrics.
    Uses hashed ChatGPT user IDs for privacy.
    """
    __tablename__ = 'user_gpt_contributions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id_hash = Column(String, unique=True, nullable=False, index=True)

    # Last recorded state for differential rewards
    last_contribution_timestamp_unix = Column(Float, nullable=True)  # Timestamp of latest conversation in last processed export
    cumulative_points_awarded = Column(Float, default=0.0, nullable=False)  # Total points rewarded so far

    # Key metrics from last processed export
    last_total_conversations = Column(Integer, default=0)
    last_total_user_messages = Column(Integer, default=0)

    times_rewarded = Column(Integer, default=0, nullable=False)  # Times reward > 0 was given
    first_contribution_at_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    latest_contribution_at_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class GPTContributionProof(Base):
    """
    Stores proof details for each GPT contribution attempt.
    """
    __tablename__ = 'gpt_contribution_proofs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id_hash = Column(String, nullable=False, index=True)

    file_id = Column(BigInteger, nullable=True)
    job_id = Column(String, nullable=True)
    owner_address = Column(String, nullable=True)  # Wallet address of contributor

    # PoC Results
    score_awarded_this_run = Column(Float, nullable=False)  # Differential score awarded for this run
    final_score_of_export = Column(Float, nullable=False)  # Total score calculated for current export

    valid = Column(Boolean, default=False)
    authenticity = Column(Float, default=0.0)
    ownership = Column(Float, default=0.0)
    quality = Column(Float, default=0.0)
    uniqueness = Column(Float, default=1.0)

    attributes_json = Column(JSON, nullable=True)  # 'attributes' dict from ProofResponse
    metadata_json = Column(JSON, nullable=True)    # 'metadata' dict from ProofResponse

    created_at_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc))