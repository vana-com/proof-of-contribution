import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from gpt_dao_proof.models.db import Base
from gpt_dao_proof.db_config import DatabaseManager

logger = logging.getLogger(__name__)

class Database:
    """Database connection and session manager for TEE environment"""

    def __init__(self):
        self._engine = None
        self._SessionLocal: Optional[sessionmaker[Session]] = None

    def _get_connection_string(self) -> str:
        try:
            return DatabaseManager.initialize_from_env()
        except ValueError as e:
            logger.error(f"Failed to initialize database connection string: {e}")
            raise

    def init(self) -> None:
        """Initialize database connection and create tables."""
        if self._engine:
            logger.info("Database already initialized.")
            return
        try:
            connection_string = self._get_connection_string()
            self._engine = create_engine(connection_string)

            inspector = inspect(self._engine)
            existing_tables = inspector.get_table_names()

            # checkfirst=True handles existing tables
            Base.metadata.create_all(self._engine, checkfirst=True)

            self._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
            logger.info("Database initialized successfully and tables ensured.")

        except SQLAlchemyError as e:
            logger.error(f"Database initialization failed: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during database initialization: {e}")
            raise


    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        if not self._SessionLocal:
            logger.error("Database not initialized. Call init() first.")
            raise RuntimeError("Database not initialized. Call init() first.")

        session = self._SessionLocal()
        try:
            yield session
            session.commit()
            logger.debug("DB Session committed.")
        except Exception as e:
            logger.error(f"DB Session error, rolling back: {e}")
            session.rollback()
            raise
        finally:
            session.close()
            logger.debug("DB Session closed.")

    def get_session(self) -> Session:
        if not self._SessionLocal:
            logger.error("Database not initialized. Call init() first.")
            raise RuntimeError("Database not initialized. Call init() first.")
        return self._SessionLocal()

    def dispose(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._SessionLocal = None
            logger.info("Database engine disposed.")

# Global database instance
db = Database()