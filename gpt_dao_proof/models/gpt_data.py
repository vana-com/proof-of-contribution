"""Domain models for handling processed ChatGPT data within the PoC."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class UserProfile:
    user_id_hash: str
    is_plus_user: bool

@dataclass
class Message:
    id: str
    role: str  # "user", "assistant", "system"
    text_content: str
    model_slug: Optional[str] = None
    timestamp_unix: Optional[float] = None
    has_code_block: bool = False

@dataclass
class Conversation:
    id: str
    title: Optional[str] = None
    create_time_unix: Optional[float] = None
    update_time_unix: Optional[float] = None
    messages: List[Message] = field(default_factory=list)
    message_count: int = 0

@dataclass
class Feedback:
    rating: str  # "thumbs_up", "thumbs_down"

@dataclass
class ProcessedGPTData:
    """Holds all processed information from the ChatGPT export."""
    user_profile: Optional[UserProfile] = None
    conversations: List[Conversation] = field(default_factory=list)
    feedback_entries: List[Feedback] = field(default_factory=list)

    # Aggregated metrics for infographic attributes
    # Volume & Activity
    interaction_galaxy_data: List[Dict[str, Any]] = field(default_factory=list)  # [{"date": "YYYY-MM-DD", "count": N}]
    total_conversations_initiated: int = 0
    total_user_prompts_penned: int = 0
    total_ai_insights_received: int = 0
    ai_journey_started_date_str: Optional[str] = None
    total_days_active: int = 0
    busiest_day_str: Optional[str] = None
    busiest_day_interactions: int = 0

    # Language & Style
    word_cloud_data: List[Dict[str, Any]] = field(default_factory=list)  # [{"text": "word", "value": count}]
    code_weaver_prompts_count: int = 0

    # Interaction Patterns & Persona
    peak_ai_synchronicity_data: Dict[str, float] = field(default_factory=dict)  # {"Morning": %, ...}
    peak_ai_synchronicity_slot_str: Optional[str] = None
    genesis_style_analytical_pct: float = 0.0
    genesis_style_creative_pct: float = 0.0

    # Conversation Dynamics
    longest_conversation_messages: int = 0
    shortest_conversation_messages: int = 0
    average_conversation_messages: float = 0.0

    # Feedback Footprint
    feedback_total_insights_shared: int = 0
    feedback_positive_count: int = 0
    feedback_improvement_suggestions_count: int = 0

    # Badge-related raw data
    gpt4_message_count: int = 0
    total_assistant_model_messages_with_slug: int = 0  # For GPT-4 usage %
    image_generation_triggers_count: int = 0
    voice_input_triggers_count: int = 0

    # Internal processing timestamps
    first_conversation_unix: Optional[float] = None
    last_conversation_unix: Optional[float] = None