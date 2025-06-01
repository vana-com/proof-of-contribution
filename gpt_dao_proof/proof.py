"""Main proof generation logic for GPT DAO."""
import hashlib
import json
import logging
import os
import traceback
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional

from sqlalchemy.orm import Session

from gpt_dao_proof.config import settings, Settings
from gpt_dao_proof.models.proof import ProofResponse, ProofMetadata, FileInfo
from gpt_dao_proof.models.gpt_data import ProcessedGPTData
from gpt_dao_proof.services.data_parser import ChatGPTDataParser
from gpt_dao_proof.services.storage import GPTStorageService
from gpt_dao_proof.scoring import GPTContributionScorer
from gpt_dao_proof.db import db

logger = logging.getLogger(__name__)

class GPTProofGenerator:
    """Handles proof generation and validation for ChatGPT data exports."""

    def __init__(self, settings_obj: Settings, db_session: Session):
        self.settings = settings_obj
        self.scorer = GPTContributionScorer()
        self.storage = GPTStorageService(db_session)


    def _extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        try:
            if not extract_to.exists():
                extract_to.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                total_uncompressed_size = 0
                num_files = 0
                MAX_TOTAL_UNCOMPRESSED_SIZE = 500 * 1024 * 1024
                MAX_INDIVIDUAL_FILE_SIZE = 100 * 1024 * 1024
                MAX_NUM_FILES = 100

                for member in zip_ref.infolist():
                    if member.filename.startswith('/') or '..' in member.filename:
                        logger.error(f"Path traversal attempt in ZIP: {member.filename}")
                        return False
                    if member.file_size > MAX_INDIVIDUAL_FILE_SIZE:
                        logger.error(f"File {member.filename} exceeds individual size limit.")
                        return False
                    total_uncompressed_size += member.file_size
                    num_files += 1
                    if total_uncompressed_size > MAX_TOTAL_UNCOMPRESSED_SIZE:
                        logger.error("Total uncompressed size of ZIP archive exceeds limit.")
                        return False
                    if num_files > MAX_NUM_FILES:
                        logger.error("Too many files in ZIP archive.")
                        return False
                zip_ref.extractall(extract_to)
            logger.info(f"Successfully extracted '{zip_path.name}' to '{extract_to}'")
            return True
        except zipfile.BadZipFile:
            logger.error(f"Invalid or corrupted ZIP file: {zip_path.name}")
            return False
        except Exception as e:
            logger.error(f"Failed to extract ZIP '{zip_path.name}': {str(e)}")
            return False

    def _validate_file_structure(self, extracted_path: Path) -> bool:
        user_json_path = extracted_path / "user.json"
        conversations_json_path = extracted_path / "conversations.json"
        chat_html_path = extracted_path / "chat.html"
        if not user_json_path.exists():
            logger.error(f"'user.json' not found in extracted data at {extracted_path}.")
            return False
        if not (conversations_json_path.exists() or chat_html_path.exists()):
            logger.error(f"Neither 'conversations.json' nor 'chat.html' found at {extracted_path}.")
            return False
        logger.info("Basic file structure validation passed.")
        return True

    def _create_attributes_from_processed_data(self, data: ProcessedGPTData) -> Dict[str, Any]:
        attrs = {}
        if data.user_profile:
            attrs['user_id_hash'] = data.user_profile.user_id_hash
            attrs['is_plus_user'] = data.user_profile.is_plus_user
        attrs.update(data.attributes)
        attrs['interaction_galaxy_data'] = data.interaction_galaxy_data
        attrs['total_conversations_initiated'] = data.total_conversations_initiated
        attrs['total_user_prompts_penned'] = data.total_user_prompts_penned
        attrs['total_ai_insights_received'] = data.total_ai_insights_received
        attrs['ai_journey_started_date'] = data.ai_journey_started_date_str
        attrs['total_days_active'] = data.total_days_active
        attrs['busiest_day_str'] = data.busiest_day_str
        attrs['busiest_day_interactions'] = data.busiest_day_interactions
        attrs['idea_galaxy_data'] = data.word_cloud_data
        attrs['code_weaver_prompts_count'] = data.code_weaver_prompts_count
        attrs['peak_ai_synchronicity_data'] = data.peak_ai_synchronicity_data
        attrs['peak_ai_synchronicity_slot_str'] = data.peak_ai_synchronicity_slot_str
        attrs['genesis_style_analytical_pct'] = data.genesis_style_analytical_pct
        attrs['genesis_style_creative_pct'] = data.genesis_style_creative_pct
        attrs['longest_conversation_messages'] = data.longest_conversation_messages
        attrs['shortest_conversation_messages'] = data.shortest_conversation_messages
        attrs['average_conversation_messages'] = data.average_conversation_messages
        attrs['feedback_total_insights_shared'] = data.feedback_total_insights_shared
        attrs['feedback_positive_count'] = data.feedback_positive_count
        attrs['feedback_improvement_suggestions_count'] = data.feedback_improvement_suggestions_count
        if data.total_assistant_model_messages_with_slug > 0:
            gpt4_usage_pct = data.gpt4_message_count / data.total_assistant_model_messages_with_slug
            attrs['badge_gpt4_power_user_flag'] = gpt4_usage_pct >= settings.BADGE_GPT4_USAGE_THRESHOLD_PCT
        else:
            attrs['badge_gpt4_power_user_flag'] = False
        attrs['badge_dalle_visionary_flag'] = data.image_generation_triggers_count > 0
        attrs['badge_voice_virtuoso_flag'] = data.voice_input_triggers_count > 0
        attrs['badge_ai_refiner_gold_flag'] = data.feedback_total_insights_shared >= settings.BADGE_FEEDBACK_GOLD_COUNT
        attrs['badge_ai_refiner_silver_flag'] = settings.BADGE_FEEDBACK_SILVER_COUNT <= data.feedback_total_insights_shared < settings.BADGE_FEEDBACK_GOLD_COUNT
        attrs['badge_ai_refiner_bronze_flag'] = settings.BADGE_FEEDBACK_BRONZE_COUNT <= data.feedback_total_insights_shared < settings.BADGE_FEEDBACK_SILVER_COUNT
        attrs['badge_consistent_conversationalist_flag'] = data.total_days_active >= settings.BADGE_CONSISTENT_DAYS_THRESHOLD
        return attrs

    def generate(self) -> ProofResponse:
        input_path = Path(self.settings.INPUT_DIR)
        temp_extract_path = input_path / "extracted_gpt_data"
        zip_file_path: Optional[Path] = None

        for item in input_path.iterdir():
            if item.is_file() and item.suffix.lower() == '.zip':
                zip_file_path = item
                break

        file_info_obj = None
        if self.settings.FILE_ID is not None and self.settings.FILE_ID > 0:
            file_info_obj = FileInfo(
                id=self.settings.FILE_ID,
                source="tee_poc_gpt_dao"
            )

        proof_metadata = ProofMetadata(
            dlp_id=self.settings.DLP_ID,
            proof_version="1.0.3",
            job_id=str(self.settings.JOB_ID) if self.settings.JOB_ID is not None else "local_job_id",
            owner_address=self.settings.OWNER_ADDRESS,
            file_info=file_info_obj
        )

        if not zip_file_path:
            logger.error("No ZIP file found in input directory.")
            return ProofResponse(dlp_id=self.settings.DLP_ID, valid=False, score=0.0,
                                 attributes={"error": "No ZIP file provided"}, metadata=proof_metadata)

        if not self._extract_zip(zip_file_path, temp_extract_path):
            return ProofResponse(dlp_id=self.settings.DLP_ID, valid=False, score=0.0,
                                 attributes={"error": f"Failed to extract ZIP file: {zip_file_path.name}"}, metadata=proof_metadata)

        if not self._validate_file_structure(temp_extract_path):
            return ProofResponse(dlp_id=self.settings.DLP_ID, valid=False, score=0.0,
                                 attributes={"error": "Invalid or incomplete file structure in ZIP."}, metadata=proof_metadata)

        try:
            parser = ChatGPTDataParser(str(temp_extract_path))
            processed_data: ProcessedGPTData = parser.parse()

            if not processed_data.user_profile or not processed_data.user_profile.user_id_hash:
                logger.error("User ID hash could not be determined from the data.")
                return ProofResponse(dlp_id=self.settings.DLP_ID, valid=False, score=0.0,
                                     attributes={"error": "User identification failed."}, metadata=proof_metadata)

            current_export_total_normalized_score, points_breakdown = self.scorer.calculate_score_and_points(processed_data)
            current_export_raw_points = points_breakdown.get('total_achieved_raw_points', 0)

            last_contribution = self.storage.get_latest_contribution(processed_data.user_profile.user_id_hash)
            previous_cumulative_rewarded_raw_points = last_contribution.cumulative_points_awarded if last_contribution else 0.0

            differential_raw_points = max(0.0, float(current_export_raw_points) - float(previous_cumulative_rewarded_raw_points))

            rewardable_score_for_contract = 0.0
            if self.settings.MAX_INFOGRAPHIC_POINTS > 0: # Denominator from config (e.g. 1000)
                rewardable_score_for_contract = min(1.0, differential_raw_points / float(self.settings.MAX_INFOGRAPHIC_POINTS))
            rewardable_score_for_contract = round(rewardable_score_for_contract, 6)


            attributes_for_response = self._create_attributes_from_processed_data(processed_data)
            attributes_for_response['points_breakdown'] = points_breakdown
            attributes_for_response['current_export_total_raw_points'] = current_export_raw_points
            attributes_for_response['previous_cumulative_rewarded_raw_points_from_db'] = previous_cumulative_rewarded_raw_points
            attributes_for_response['differential_raw_points_calculated'] = differential_raw_points
            attributes_for_response['current_export_total_normalized_score'] = current_export_total_normalized_score
            attributes_for_response['rewardable_score_for_contract'] = rewardable_score_for_contract
            is_first_contribution = not bool(last_contribution)
            attributes_for_response['is_first_contribution'] = is_first_contribution

            quality_score_val = round(current_export_total_normalized_score * 0.8 + 0.2, 4) if current_export_total_normalized_score > 0 else 0.0
            uniqueness_score_val = 1.0 if is_first_contribution else 0.95


            final_proof_response = ProofResponse(
                dlp_id=self.settings.DLP_ID,
                valid=True,
                score=rewardable_score_for_contract,
                authenticity=1.0,
                ownership=1.0,
                quality=min(1.0, quality_score_val),
                uniqueness=uniqueness_score_val,
                attributes=attributes_for_response,
                metadata=proof_metadata
            )

            self.storage.record_proof_attempt(
                proof_response_model=final_proof_response,
                current_export_total_normalized_score=current_export_total_normalized_score
            )

            self.storage.update_user_cumulative_state(
                user_id_hash=processed_data.user_profile.user_id_hash,
                differential_raw_points_to_add=differential_raw_points,
                current_export_last_timestamp_unix=processed_data.last_conversation_unix,
                current_export_metrics_snapshot={
                    "total_conversations": processed_data.total_conversations_initiated,
                    "total_user_messages": processed_data.total_user_prompts_penned,
                    "total_ai_messages": processed_data.total_ai_insights_received,
                    "gpt4_message_count": processed_data.gpt4_message_count,
                    "feedback_count": processed_data.feedback_total_insights_shared,
                    "total_days_active": processed_data.total_days_active,
                }
            )

            return final_proof_response

        except ValueError as ve:
            logger.error(f"ValueError during proof generation pipeline: {str(ve)}")
            return ProofResponse(dlp_id=self.settings.DLP_ID, valid=False, score=0.0,
                                 attributes={"error": f"Data processing error: {str(ve)}"},
                                 metadata=proof_metadata)
        except Exception as e:
            logger.error(f"Unexpected error during proof generation pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            return ProofResponse(dlp_id=self.settings.DLP_ID, valid=False, score=0.0,
                                 attributes={"error": f"Unexpected pipeline error: {str(e)}"},
                                 metadata=proof_metadata)