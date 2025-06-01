"""Database storage service for GPT DAO contributions and proofs."""
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from gpt_dao_proof.models.db import UserGPTContribution, GPTContributionProof
from gpt_dao_proof.models.proof import ProofResponse

logger = logging.getLogger(__name__)

class GPTStorageService:
    """Handles database operations for GPT DAO contributions."""

    def __init__(self, session: Session):
        self.session = session

    def get_latest_contribution(self, user_id_hash: str) -> Optional[UserGPTContribution]:
        try:
            contribution = (
                self.session.query(UserGPTContribution)
                .filter_by(user_id_hash=user_id_hash)
                .first()
            )
            if contribution:
                logger.info(f"Found existing contribution record for user_id_hash: {user_id_hash} with cumulative_points_awarded: {contribution.cumulative_points_awarded}")
            else:
                logger.info(f"No existing contribution record found for user_id_hash: {user_id_hash}")
            return contribution
        except SQLAlchemyError as e:
            logger.error(f"DB error fetching latest contribution for {user_id_hash}: {e}")
            raise

    def update_user_cumulative_state(
            self,
            user_id_hash: str,
            differential_raw_points_to_add: float,
            current_export_last_timestamp_unix: Optional[float],
            current_export_metrics_snapshot: Dict[str, Any]
    ) -> UserGPTContribution:
        """
        Updates the user's cumulative rewarded points and last processed state.
        This is called by the PoC after it has calculated the differential points for the current run
        and is about to output a score based on these differential points.
        """
        try:
            contribution = self.get_latest_contribution(user_id_hash)
            now_utc = datetime.now(timezone.utc)

            if contribution:
                logger.info(f"Updating UserGPTContribution cumulative state for hash: {user_id_hash}")
                if differential_raw_points_to_add > 0:
                    contribution.cumulative_points_awarded += differential_raw_points_to_add
                    contribution.times_rewarded += 1
                contribution.last_contribution_timestamp_unix = current_export_last_timestamp_unix
                contribution.latest_contribution_at_utc = now_utc
                contribution.last_total_conversations = current_export_metrics_snapshot.get("total_conversations", contribution.last_total_conversations)
                contribution.last_total_user_messages = current_export_metrics_snapshot.get("total_user_messages", contribution.last_total_user_messages)
            else:
                logger.info(f"Creating new UserGPTContribution cumulative state for hash: {user_id_hash}")
                contribution = UserGPTContribution(
                    user_id_hash=user_id_hash,
                    cumulative_points_awarded=differential_raw_points_to_add,
                    times_rewarded=1 if differential_raw_points_to_add > 0 else 0,
                    last_contribution_timestamp_unix=current_export_last_timestamp_unix,
                    latest_contribution_at_utc=now_utc,
                    first_contribution_at_utc=now_utc,
                    last_total_conversations=current_export_metrics_snapshot.get("total_conversations", 0),
                    last_total_user_messages=current_export_metrics_snapshot.get("total_user_messages", 0)
                )
                self.session.add(contribution)
            logger.info(f"UserGPTContribution cumulative state for {user_id_hash} updated. New cumulative_points_awarded: {contribution.cumulative_points_awarded}. DB commit handled by caller.")
            return contribution
        except SQLAlchemyError as e:
            logger.error(f"DB error updating user cumulative state for {user_id_hash}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating user cumulative state for {user_id_hash}: {e}")
            raise

    def record_proof_attempt(self, proof_response_model: ProofResponse, current_export_total_normalized_score: float):
        """
        Records the details of this specific proof generation attempt.
        'proof_response_model.score' is the normalized differential score (for the contract).
        'current_export_total_normalized_score' is the total normalized score of the current export file.
        """
        try:
            user_id_hash = proof_response_model.attributes.get("user_id_hash")
            if not user_id_hash:
                logger.error("Cannot record proof attempt: user_id_hash missing from attributes.")
                return

            metadata_dump = {}
            file_id_for_db = None
            if proof_response_model.metadata:
                metadata_dump = proof_response_model.metadata.model_dump(exclude_none=True) # Use exclude_none
                if proof_response_model.metadata.file_info:
                    file_id_for_db = proof_response_model.metadata.file_info.id


            proof_record = GPTContributionProof(
                user_id_hash=user_id_hash,
                file_id=file_id_for_db,
                job_id=str(proof_response_model.metadata.job_id) if proof_response_model.metadata else None,
                owner_address=str(proof_response_model.metadata.owner_address) if proof_response_model.metadata else None,

                score_awarded_this_run=proof_response_model.score,
                final_score_of_export=current_export_total_normalized_score,

                valid=proof_response_model.valid,
                authenticity=proof_response_model.authenticity,
                ownership=proof_response_model.ownership,
                quality=proof_response_model.quality,
                uniqueness=proof_response_model.uniqueness,
                attributes_json=proof_response_model.attributes,
                metadata_json=metadata_dump
            )
            self.session.add(proof_record)
            logger.info(f"GPTContributionProof record prepared for user_id_hash: {user_id_hash}, job_id: {proof_response_model.metadata.job_id if proof_response_model.metadata else 'N/A'}")
            logger.info(f"Proof attempt details: score_awarded_this_run={proof_response_model.score}, final_score_of_export={current_export_total_normalized_score}")

        except SQLAlchemyError as e:
            logger.error(f"DB error recording proof attempt: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error recording proof attempt: {e}")
            raise