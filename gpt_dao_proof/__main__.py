"""Entry point for GPT DAO proof of contribution generation."""
import json
import logging
import os
import sys
import traceback
import zipfile
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from typing import Optional

from gpt_dao_proof.config import settings
from gpt_dao_proof.proof import GPTProofGenerator
from gpt_dao_proof.db import db

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_poc() -> None:
    """
    Main function to orchestrate the Proof of Contribution generation.
    Initializes database, runs the proof generator, and handles output.
    Manages the database session lifecycle for the PoC run.
    """
    db_session: Optional[Session] = None

    try:
        db.init()
        logger.info("Database connection pool initialized.")

        db_session = db.get_session()
        logger.info("Database session acquired.")

        input_path = Path(settings.INPUT_DIR)
        if not input_path.is_dir() or not any(input_path.iterdir()):
            logger.error(f"No input files found in {settings.INPUT_DIR} or directory does not exist.")

        logger.info("GPT DAO Proof of Contribution - TEE Run Starting")
        logger.info("Settings (sensitive fields excluded):")
        safe_config = settings.model_dump(exclude={'DB_PASSWORD'})
        logger.info(json.dumps(safe_config, indent=2))

        proof_generator = GPTProofGenerator(settings, db_session)
        proof_response = proof_generator.generate()

        if proof_response.valid:
            db_session.commit()
            logger.info("Database session committed successfully after proof generation.")
        else:
            # Commit even for "valid=False" proofs to save the attempt record
            db_session.commit()
            logger.info("Database session committed for 'valid=False' proof (e.g., to save attempt record).")


        output_dir_path = Path(settings.OUTPUT_DIR)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True, exist_ok=True)

        output_file_path = output_dir_path / "results.json"
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(proof_response.model_dump(), f, indent=2)

        logger.info(f"Proof generation complete. Results saved to {output_file_path}")
        if proof_response.valid:
            logger.info(f"Proof Score: {proof_response.score}")
        else:
            logger.warning(f"Proof generation resulted in an invalid proof. Error: {proof_response.attributes.get('error', 'Unknown error')}")

    except Exception as e:
        logger.critical(f"CRITICAL: Unhandled error during proof generation: {str(e)}")
        logger.critical(traceback.format_exc())
        if db_session:
            try:
                db_session.rollback()
                logger.info("Database session rolled back due to critical error.")
            except Exception as rb_err:
                logger.error(f"Error during session rollback: {rb_err}")

        error_response_dict = {
            "dlp_id": settings.DLP_ID, "valid": False, "score": 0.0,
            "attributes": {"error": f"Unhandled Exception: {str(e)}", "traceback": traceback.format_exc()},
            "metadata": {"proof_version": "1.0.1", "job_id": str(settings.JOB_ID)}
        }
        output_dir_path = Path(settings.OUTPUT_DIR)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir_path / "results.json"
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(error_response_dict, f, indent=2)
            logger.info(f"Error results due to unhandled exception saved to {output_file_path}")
        except Exception as write_err:
            logger.error(f"Additionally failed to write error results to output file: {write_err}")
        raise # Re-raise the exception after attempting to log and write error output
    finally:
        if db_session:
            try:
                db_session.close()
                logger.info("Database session closed.")
            except Exception as close_err:
                logger.error(f"Error closing database session: {close_err}")

        if 'db' in globals() and hasattr(db, 'dispose') and db._engine is not None:
            logger.info("Disposing database connection pool...")
            db.dispose()
            logger.info("Database connection pool disposed.")
        else:
            logger.info("Database was not initialized or already disposed, skipping pool dispose.")
        logger.info("GPT DAO Proof of Contribution - TEE Run Finished.")

if __name__ == "__main__":
    logger.info("Running GPT DAO Proof of Contribution locally for testing/development...")

    # Uncomment below to use test data instead of run_poc()
    # run_poc()
    # exit(0)

    run_poc()
    exit(0)

    # Test setup code below (commented out for production)
    Path(settings.INPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.DEBUG_DIR).mkdir(parents=True, exist_ok=True)

    # Create dummy test data
    dummy_zip_filename = "gpt_export_dummy_local.zip"
    dummy_zip_path = Path(settings.INPUT_DIR) / dummy_zip_filename

    if not dummy_zip_path.exists():
        logger.info(f"Creating dummy ZIP file for local testing at {dummy_zip_path}...")
        dummy_user_json_content = {"id": "user-local-test-001", "chatgpt_plus_user": True,
                                   "email": "localtest@example.com"}

        ts1_create = datetime(2023, 1, 10, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        ts1_update = datetime(2023, 1, 10, 12, 5, 0, tzinfo=timezone.utc).timestamp()
        ts2_create = datetime(2023, 2, 15, 18, 30, 0, tzinfo=timezone.utc).timestamp()
        ts2_update = datetime(2023, 2, 15, 18, 40, 0, tzinfo=timezone.utc).timestamp()

        dummy_conversations_for_html = [
            {
                "title": "First Local Test", "create_time": ts1_create, "update_time": ts1_update, "id": "conv1",
                "mapping": {
                    "msgA": {"id": "msgA",
                             "message": {"author": {"role": "user"}, "content": {"parts": ["Hello AI, how are you?"]},
                                         "metadata": {}, "create_time": ts1_create}},
                    "msgB": {"id": "msgB", "message": {"author": {"role": "assistant"},
                                                       "content": {"parts": ["I am well, thank you!"]},
                                                       "metadata": {"model_slug": "gpt-3.5-turbo"}},
                             "create_time": ts1_create + 60}}
            },
            {
                "title": "Coding Question", "create_time": ts2_create, "update_time": ts2_update, "id": "conv2",
                "mapping": {
                    "msgC": {"id": "msgC", "message": {"author": {"role": "user"}, "content": {
                        "parts": ["Write python code ```python\nprint('hello world')\n``` this is a test."]},
                                                       "metadata": {}, "create_time": ts2_create}},
                    "msgD": {"id": "msgD", "message": {"author": {"role": "assistant"}, "content": {
                        "parts": ["Okay, here's the Python code: ```python\nprint('Hello World!')\n```"]},
                                                       "metadata": {"model_slug": "gpt-4"}},
                             "create_time": ts2_create + 60}}
            }
        ]
        dummy_chat_html_content = f"<html><head><script>var jsonData = {json.dumps(dummy_conversations_for_html)};</script></head><body>Chat content here</body></html>"
        dummy_feedback_json_content = [
            {"conversation_id": "conv1", "user_id": "user-local-test-001", "rating": "thumbs_up",
             "create_time": ts1_update + 120}
        ]

        try:
            with zipfile.ZipFile(dummy_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("user.json", json.dumps(dummy_user_json_content))
                zf.writestr("chat.html", dummy_chat_html_content)
                zf.writestr("message_feedback.json", json.dumps(dummy_feedback_json_content))
            logger.info(f"Dummy ZIP file '{dummy_zip_filename}' created successfully in {settings.INPUT_DIR}.")
        except Exception as e_zip:
            logger.error(f"Failed to create dummy ZIP for local testing: {e_zip}")
    else:
        logger.info(f"Found existing test ZIP: {dummy_zip_path}. Using it.")

    # Execute the PoC
    exit_code = 0
    try:
        run_poc()
    except Exception:
        exit_code = 1

    sys.exit(exit_code)
