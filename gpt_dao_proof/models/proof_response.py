from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from gpt_dao_proof.models.proof import ProofMetadata # Import shared ProofMetadata


class ProofResponse(BaseModel):
    """
    Represents the response of a proof of contribution.
    The 'score' field is the critical output for the DLP smart contract,
    representing the normalized differential value of the current contribution.
    'attributes' contains all detailed metrics from the current export for off-chain use (e.g., infographic).
    'metadata' includes details about the PoC run itself.
    """

    dlp_id: int
    valid: bool = Field(default=False, description="Overall validity of the contribution based on PoC checks.")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized differential score (0-1) for this contribution, representing new value. This is the score the smart contract will use for rewards.")
    authenticity: float = Field(default=1.0, ge=0.0, le=1.0, description="Score rating if data is authentic (e.g., from a real export). Currently hardcoded for V0.")
    ownership: float = Field(default=1.0, ge=0.0, le=1.0, description="Score verifying file ownership. Currently hardcoded for V0.")
    quality: float = Field(default=0.0, ge=0.0, le=1.0, description="Score based on the overall richness, model usage, feedback, etc. of the current export file.")
    uniqueness: float = Field(default=1.0, ge=0.0, le=1.0, description="Score indicating data novelty (e.g., 1.0 for first contribution).")

    attributes: Dict[str, Any] = Field(default_factory=dict, description="Detailed metrics from the current export for off-chain use (e.g., infographic generation, analytics). Includes calculated raw points, differential points, and normalized scores.")
    metadata: ProofMetadata

    class Config:
        extra = 'allow'