"""ProofResponse model definition, tailored for GPT DAO."""
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

class FileInfo(BaseModel):
    id: int = Field(description="File ID")
    source: str = Field(description="Source of file generation (e.g., 'tee')")
    url: Optional[str] = Field(None, description="File URL in storage (if applicable)")
    checksums: Optional[Dict[str, str]] = Field(None, description="File checksums (if applicable)")

class ProofMetadata(BaseModel):
    dlp_id: int = Field(description="Data Liquidity Pool ID")
    proof_version: str = Field(description="Proof of Contribution version")
    job_id: Optional[str] = Field(None, description="TEE job ID")
    owner_address: Optional[str] = Field(None, description="Owner's wallet address")
    file_info: Optional[FileInfo] = Field(None, description="File information (if applicable)")

class ProofResponse(BaseModel):
    dlp_id: int
    valid: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall score (0-1) for this contribution export")
    authenticity: float = Field(default=0.0, ge=0.0, le=1.0, description="Score rating if data is authentic (e.g., from a real export)")
    ownership: float = Field(default=0.0, ge=0.0, le=1.0, description="Score verifying file ownership (implicit for user uploads)")
    quality: float = Field(default=0.0, ge=0.0, le=1.0, description="Score based on richness, model usage, feedback, etc.")
    uniqueness: float = Field(default=1.0, ge=0.0, le=1.0, description="Score indicating data novelty")

    attributes: Dict[str, Any] = Field(default_factory=dict, description="Detailed metrics for infographic and DAO use")
    metadata: ProofMetadata

    class Config:
        extra = 'allow'  # Allow arbitrary types in attributes