"""Database configuration and credentials management for TEE"""
from dataclasses import dataclass

from gpt_dao_proof.config import settings

# Network-specific database configurations
MAINNET_CONFIG = {
    'HOST': 'ep-cool-morning-a12p7v4u-pooler.ap-southeast-1.aws.neon.tech',
    'PORT': '5432',
    'NAME': 'gptdatadao_mainnet',
    'USER': 'neondb_owner',
    'SSL_MODE': 'require'
}

TESTNET_CONFIG = {
    'HOST': 'ep-cool-morning-a12p7v4u-pooler.ap-southeast-1.aws.neon.tech',
    'PORT': '5432',
    'NAME': 'gptdatadao_testent',
    'USER': 'neondb_owner',
    'SSL_MODE': 'require'
}

LOCAL_CONFIG = {
    'HOST': 'ep-cool-morning-a12p7v4u-pooler.ap-southeast-1.aws.neon.tech',
    'PORT': '5432',
    'NAME': 'gptdatadao_testent',
    'USER': 'neondb_owner',
    'SSL_MODE': 'require'
}

def determine_network_config() -> dict:
    """Determine database configuration based on DLP_ID from settings."""
    if settings.DLP_ID == 32:  # Mainnet
        return MAINNET_CONFIG
    elif settings.DLP_ID == 89:  # Testnet
        return TESTNET_CONFIG
    elif settings.DLP_ID == 0:  # Local Dev
        return LOCAL_CONFIG
    else:
        print(f"Warning: Unrecognized DLP_ID {settings.DLP_ID}. Falling back to LOCAL_CONFIG for DB.")
        return LOCAL_CONFIG

DB_CONFIG = determine_network_config()

@dataclass
class DatabaseCredentials:
    """Database credentials container."""
    host: str
    port: str
    name: str
    user: str
    password: str
    ssl_mode: str = 'require'

    def to_connection_string(self) -> str:
        """Generate database connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}?sslmode={self.ssl_mode}"

    @classmethod
    def from_settings(cls) -> 'DatabaseCredentials':
        """Create credentials from global DB_CONFIG and settings.DB_PASSWORD."""
        if not settings.DB_PASSWORD:
            raise ValueError("DB_PASSWORD setting is required and not found.")
        return cls(
            host=DB_CONFIG['HOST'],
            port=DB_CONFIG['PORT'],
            name=DB_CONFIG['NAME'],
            user=DB_CONFIG['USER'],
            password=settings.DB_PASSWORD,
            ssl_mode=DB_CONFIG['SSL_MODE']
        )

class DatabaseManager:
    """Manages database connection string generation."""

    @staticmethod
    def initialize_from_env() -> str:
        """
        Initialize database connection string from settings.
        Returns: Database connection string
        Raises: ValueError if required settings are missing
        """
        credentials = DatabaseCredentials.from_settings()
        return credentials.to_connection_string()

