import hashlib
import json
import os
import re
import logging
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from gpt_dao_proof.models.gpt_data import (
    ProcessedGPTData,
    UserProfile,
    Conversation,
    Message,
    Feedback
)
from gpt_dao_proof.config import settings

logger = logging.getLogger(__name__)

# --- Stop Words ---
# TODO: Use langetect package or another ML model to detect languale, move words list bellow by language to distinct location
STOP_WORDS = set([
    "a", "an", "and", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should", "can",
    "could", "may", "might", "must", "am", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "mine", "yours", "hers", "ours", "theirs", "myself", "yourself",
    "himself", "herself", "itself", "ourselves", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "but", "if", "or", "as", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "im", "ive", "id", "chatgpt",
    "ll", "re", "ve", "m", "o", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn", "one", "like", "get", "make", "use", "see",
    "know", "tell", "ask", "give", "explain", "example", "examples", "way", "ways",
    "please", "thank", "thanks", "sure", "okay", "yes", "help", "new", "also", "hi", "hello",
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "actual", "actually", "think", "could", "can", "one", "something", "thing", "much", "many", "lot",
    "things", "good", "well", "really", "bit", "mean", "say", "make", "go", "want", "need", "even", "just", "right",
    "let", "know", "try", "kind", "part", "look", "find", "way", "new", "used", "using", "able", "sure", "give",
    "take", "set", "put", "keep", "ask", "talk", "long", "create", "provide", "generate"
])
ANALYTICAL_KEYWORDS = ["code", "debug", "algorithm", "data", "analyze", "analysis", "logic", "optimize", "database", "security", "test", "deploy", "python", "javascript", "react", "api", "machine learning", "calculate", "technical", "system", "architecture", "framework", "function", "class", "variable", "error", "bug", "exception", "performance", "query", "index", "network", "server", "cloud", "docker", "kubernetes", "terraform", "aws", "azure", "gcp", "linux", "windows", "macos", "shell", "bash", "powershell", "git", "version control", "ci/cd", "automation", "script", "json", "xml", "yaml", "csv", "excel", "sql", "nosql", "api", "rest", "graphql", "grpc", "http", "tcp", "ip", "dns", "ssl", "tls", "encryption", "decryption", "authentication", "authorization", "oauth", "jwt", "saml", "openid", "kerberos", "ldap", "active directory", "firewall", "vpn", "proxy", "load balancer", "cache", "cdn", "monitoring", "logging", "alerting", "metrics", "tracing", "observability", "devops", "sre", "agile", "scrum", "kanban", "jira", "confluence", "slack", "teams", "zoom", "webex", "gotomeeting", "skype", "discord", "figma", "adobe xd", "sketch", "invision", "zeplin", "miro", "mural", "trello", "asana", "monday", "notion", "airtable", "salesforce", "hubspot", "marketo", "pardot", "google analytics", "google tag manager", "google data studio", "google bigquery", "google cloud storage", "amazon s3", "amazon ec2", "amazon rds", "amazon dynamodb", "amazon lambda", "amazon api gateway", "amazon cognito", "amazon sagemaker", "amazon comprehend", "amazon rekognition", "amazon translate", "amazon polly", "amazon lex", "amazon connect", "microsoft azure virtual machines", "microsoft azure blob storage", "microsoft azure sql database", "microsoft azure cosmos db", "microsoft azure functions", "microsoft azure api management", "microsoft azure active directory", "microsoft azure machine learning", "microsoft azure cognitive services", "google cloud compute engine", "google cloud storage", "google cloud sql", "google cloud spanner", "google cloud functions", "google cloud endpoints", "google cloud identity platform", "google cloud ai platform", "google cloud natural language api", "google cloud vision api", "google cloud speech-to-text api", "google cloud text-to-speech api", "google cloud dialogflow"]
CREATIVE_KEYWORDS = ["story", "poem", "write", "script", "character", "imagine", "creative", "design", "idea", "brainstorm", "narrative", "visual", "concept", "art", "music", "dialogue", "novel", "plot", "scene", "setting", "theme", "tone", "style", "voice", "song", "lyrics", "melody", "harmony", "rhythm", "beat", "instrument", "genre", "band", "album", "track", "single", "ep", "live", "concert", "tour", "festival", "gallery", "exhibition", "museum", "sculpture", "painting", "drawing", "photograph", "film", "movie", "tv show", "series", "episode", "actor", "actress", "director", "producer", "writer", "editor", "cinematographer", "composer", "sound designer", "costume designer", "set designer", "makeup artist", "hairstylist", "choreographer", "dancer", "singer", "musician", "artist", "designer", "architect", "fashion designer", "interior designer", "graphic designer", "web designer", "game designer", "illustrator", "animator", "cartoonist", "comedian", "magician", "juggler", "clown", "mime", "puppeteer", "ventriloquist", "storyteller", "poet", "novelist", "playwright", "screenwriter", "journalist", "blogger", "vlogger", "podcaster", "influencer", "youtuber", "instagrammer", "tiktoker", "twitch streamer"]
IMAGE_KEYWORDS = ["dall-e", "generate image", "create image", "draw picture", "visualize", "stable diffusion", "midjourney", "show me an image", "picture of", "photo of", "illustration of", "art of", "render an image"]
VOICE_KEYWORDS = ["speak", "say", "voice", "pronounce", "read this aloud", "in a [adj] voice", "narrate", "speech synthesis"]


class ChatGPTDataParser:
    def __init__(self, extracted_data_path: str):
        self.base_path = Path(extracted_data_path)
        self.processed_data = ProcessedGPTData()
        self.raw_user_messages_text: List[str] = []
        self.raw_assistant_messages_text: List[str] = []
        logger.info(f"DataParser initialized for path: {self.base_path}")

    def _load_json_file(self, filename: str) -> Any:
        file_path = self.base_path / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    logger.debug(f"Successfully loaded {filename}")
                    return content
            except json.JSONDecodeError as e:
                logger.warning(f"Could not decode JSON from {filename}: {e}")
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
        else:
            logger.warning(f"{filename} not found at {file_path}")
        return None

    def _parse_conversations_from_html(self) -> List[Dict]:
        html_path = self.base_path / "chat.html"
        if not html_path.exists():
            logger.warning("chat.html not found.")
            return []
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # More robust regex to find the jsonData array, handling potential variations
            match = re.search(r'var\s+jsonData\s*=\s*(\[[\s\S]*?\])\s*;', html_content, re.DOTALL)
            if not match: # Try without 'var' and semicolon, or if it's embedded differently
                match = re.search(r'jsonData\s*=\s*(\[[\s\S]*?\])', html_content, re.DOTALL)
            if not match: # Try to find just the array itself as a last resort
                match = re.search(r'(\[{"title":.*?mapping":.*?}\])', html_content, re.DOTALL)


            if match:
                json_str = match.group(1)
                # Attempt to clean potential JS issues like trailing commas before parsing
                json_str = re.sub(r',\s*([\}\]])', r'\1', json_str) # Remove trailing commas before ] or }
                json_str = json_str.strip()
                logger.debug(f"Extracted jsonData string (first 200 chars): {json_str[:200]}")
                parsed_json = json.loads(json_str)
                logger.info(f"Successfully parsed jsonData from chat.html, found {len(parsed_json)} conversations.")
                return parsed_json
            else:
                logger.warning("Could not find jsonData variable in chat.html.")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from chat.html: {str(e)}. Content snippet: {html_content[:500]}") # Log snippet for debugging
            return []
        except Exception as e:
            logger.error(f"Generic error parsing chat.html: {str(e)}")
            return []

    def _validate_conversations_structure(self, conversations_json: List[Dict]) -> None:
        """Validate the structure of conversations data and raise ValueError if invalid."""
        for i, conversation in enumerate(conversations_json):
            try:
                # Check basic conversation structure
                if not isinstance(conversation, dict):
                    raise ValueError(f"Conversation {i} is not a dictionary")
                
                # Check for mapping field
                messages_dict = conversation.get("mapping")
                if not isinstance(messages_dict, dict):
                    raise ValueError(f"Conversation {i} has invalid or missing 'mapping' field")
                
                """
                This check checks for a few of the fields included by chatgpt (but is a non-exhaustive list).
                This acts as a big filter, and detects a lot of the spam data (when I run it on a small subsample 5859/7802 fail this check)
                """
                for c in ['memory_scope', 'disabled_tool_ids', 'moderation_results', 'blocked_urls']:
                    if c not in conversation.keys():
                        raise KeyError(f"Conversation missing '{c}' fields")
                
                if not messages_dict:
                    continue  # Empty mapping is allowed
                
                # Check for root node (parent is None)
                root_candidates = [
                    v["id"]
                    for v in messages_dict.values()
                    if isinstance(v, dict) and "parent" in v and v["parent"] is None
                ]
                
                if not root_candidates:
                    raise ValueError(f"Conversation {i} has no root node (no node with parent=None)")
                
                root = root_candidates[0]
                
                # Verify root node has required structure
                if root not in messages_dict:
                    raise ValueError(f"Conversation {i} root node {root} not found in mapping")
                
                root_node = messages_dict[root]
                if not isinstance(root_node, dict) or "children" not in root_node:
                    raise ValueError(f"Conversation {i} root node missing 'children' field")
                
                # Validate message nodes have required fields
                for node_id, node_data in messages_dict.items():
                    if not isinstance(node_data, dict):
                        continue
                    
                    # Check for required node structure
                    if "id" not in node_data:
                        raise ValueError(f"Conversation {i} node {node_id} missing 'id' field")
                    
                    # If node has a message, validate message structure
                    message = node_data.get("message")
                    if message is not None:
                        if not isinstance(message, dict):
                            raise ValueError(f"Conversation {i} node {node_id} has invalid message field")
                        
                        # Check for author role
                        author = message.get("author", {})
                        if not isinstance(author, dict) or "role" not in author:
                            raise ValueError(f"Conversation {i} node {node_id} message missing author.role")
                        
                        # Check for content structure
                        content = message.get("content")
                        if content is not None and not isinstance(content, dict):
                            raise ValueError(f"Conversation {i} node {node_id} message has invalid content field")
                
            except (KeyError, TypeError, AttributeError, IndexError) as e:
                raise ValueError(f"Conversation {i} structure validation failed: {str(e)}")
        
        logger.info(f"Conversation structure validation passed for {len(conversations_json)} conversations")

    def _get_message_text_from_node(self, node_data: Optional[Dict]) -> str:
        if not node_data: return ""
        message_content = node_data.get("message", {}).get("content", {})
        parts = message_content.get("parts", [])
        if parts and isinstance(parts, list):
            # Filter out any non-string parts or empty strings more carefully
            return " ".join(str(p).strip() for p in parts if isinstance(p, str) and str(p).strip()).strip()
        return ""

    def _process_user_data(self):
        user_json = self._load_json_file("user.json")
        if user_json and user_json.get('id'):
            self.processed_data.user_profile = UserProfile(
                user_id_hash=hashlib.sha256(user_json['id'].encode('utf-8')).hexdigest(),
                is_plus_user=user_json.get('chatgpt_plus_user', False)
            )
            logger.info(f"User profile processed. ID hash: {self.processed_data.user_profile.user_id_hash}")
        else:
            logger.error("User data (user.json or ID) is missing or invalid. Proof will be marked invalid.")
            # This should ideally make the PoC invalid. This is handled in the main proof.py

    def _process_conversations(self):
        conversations_json = self._load_json_file("conversations.json")
        if not conversations_json:
            logger.info("conversations.json not found or empty, attempting to parse chat.html.")
            conversations_json = self._parse_conversations_from_html()

        if not conversations_json:
            logger.warning("No conversation data found. Subsequent metrics will be zero.")
            return

        # Validate conversation structure before processing
        self._validate_conversations_structure(conversations_json)

        daily_counts = defaultdict(int)
        hourly_counts = defaultdict(int) # For 24-hour cycle
        first_ts = float('inf')
        last_ts = float('-inf')

        for i, convo_data in enumerate(conversations_json):
            messages_in_convo_list: List[Message] = []
            convo_create_ts = convo_data.get('create_time')
            convo_update_ts = convo_data.get('update_time', convo_create_ts)

            if convo_create_ts:
                first_ts = min(first_ts, float(convo_create_ts))
                dt_obj = datetime.fromtimestamp(float(convo_create_ts), tz=timezone.utc)
                daily_counts[dt_obj.date()] += 1
                hourly_counts[dt_obj.hour] += 1 # Store hour (0-23)
            if convo_update_ts:
                last_ts = max(last_ts, float(convo_update_ts))

            mapping = convo_data.get("mapping", {})
            if not mapping: continue

            sorted_message_nodes = []
            for node_id, node_data_val in mapping.items():
                if node_data_val and node_data_val.get("message") and node_data_val["message"].get("create_time"):
                    sorted_message_nodes.append(node_data_val)

            if sorted_message_nodes:
                sorted_message_nodes.sort(key=lambda x: x["message"]["create_time"] or 0) # Handle potential None create_time
            else: # Fallback if no message-level create_time
                sorted_message_nodes = [mapping[node_id] for node_id in mapping if mapping.get(node_id, {}).get("message")]

            current_code_blocks_in_convo = 0
            for node_idx, node_data in enumerate(sorted_message_nodes):
                message_obj = node_data.get("message")
                if not message_obj: continue

                role = message_obj.get("author", {}).get("role", "unknown")
                text_content = self._get_message_text_from_node(node_data)
                model_slug = message_obj.get("metadata", {}).get("model_slug")
                msg_timestamp_unix = message_obj.get("create_time")

                has_code = "```" in text_content
                if has_code:
                    current_code_blocks_in_convo +=1

                messages_in_convo_list.append(Message(
                    id=node_data.get("id", f"msg_{i}_{node_idx}"),
                    role=role,
                    text_content=text_content,
                    model_slug=model_slug,
                    timestamp_unix=float(msg_timestamp_unix) if msg_timestamp_unix else None,
                    has_code_block=has_code
                ))

                if role == 'user':
                    self.raw_user_messages_text.append(text_content)
                    if any(keyword in text_content.lower() for keyword in IMAGE_KEYWORDS):
                        self.processed_data.image_generation_triggers_count += 1
                elif role == 'assistant':
                    self.raw_assistant_messages_text.append(text_content)
                    if model_slug:
                        self.processed_data.total_assistant_model_messages_with_slug += 1
                        if "gpt-4" in model_slug.lower():
                            self.processed_data.gpt4_message_count += 1

            if current_code_blocks_in_convo > 0: # If any message in convo had code
                self.processed_data.code_weaver_prompts_count +=1 # Count convos with code

            if messages_in_convo_list:
                self.processed_data.conversations.append(Conversation(
                    id=convo_data.get("id", f"convo_{i}"), # Use "id" field for conversation ID
                    title=convo_data.get("title"),
                    create_time_unix=float(convo_create_ts) if convo_create_ts else None,
                    update_time_unix=float(convo_update_ts) if convo_update_ts else None,
                    messages=messages_in_convo_list,
                    message_count=len(messages_in_convo_list)
                ))

        self.processed_data.total_conversations_initiated = len(self.processed_data.conversations)
        self.processed_data.total_user_prompts_penned = len(self.raw_user_messages_text)
        self.processed_data.total_ai_insights_received = len(self.raw_assistant_messages_text)

        if first_ts != float('inf'):
            self.processed_data.first_conversation_unix = first_ts
            self.processed_data.ai_journey_started_date_str = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%b %Y")
        if last_ts != float('-inf'):
            self.processed_data.last_conversation_unix = last_ts

        if self.processed_data.first_conversation_unix and self.processed_data.last_conversation_unix:
            self.processed_data.total_days_active = (datetime.fromtimestamp(self.processed_data.last_conversation_unix, tz=timezone.utc) -
                                                     datetime.fromtimestamp(self.processed_data.first_conversation_unix, tz=timezone.utc)).days + 1

        self.processed_data.interaction_galaxy_data = [{"date": date_obj.strftime('%Y-%m-%d'), "count": count} for date_obj, count in sorted(daily_counts.items())]
        if daily_counts:
            busiest_day_obj = max(daily_counts, key=daily_counts.get)
            self.processed_data.busiest_day_str = busiest_day_obj.strftime("%B %d, %Y") # e.g., March 15, 2024
            self.processed_data.busiest_day_interactions = daily_counts[busiest_day_obj]

        total_hourly_interactions = sum(hourly_counts.values())
        if total_hourly_interactions > 0:
            self.processed_data.peak_ai_synchronicity_data = {
                "Morning": round(sum(count for h, count in hourly_counts.items() if 6 <= h <= 11) / total_hourly_interactions * 100, 1),
                "Afternoon": round(sum(count for h, count in hourly_counts.items() if 12 <= h <= 17) / total_hourly_interactions * 100, 1),
                "Evening": round(sum(count for h, count in hourly_counts.items() if 18 <= h <= 23) / total_hourly_interactions * 100, 1),
                "Night": round(sum(count for h, count in hourly_counts.items() if 0 <= h <= 5) / total_hourly_interactions * 100, 1),
            }
            self.processed_data.peak_ai_synchronicity_slot_str = max(self.processed_data.peak_ai_synchronicity_data, key=self.processed_data.peak_ai_synchronicity_data.get)
        else:
            self.processed_data.peak_ai_synchronicity_data = {"Morning":0.0,"Afternoon":0.0,"Evening":0.0,"Night":0.0}
            self.processed_data.peak_ai_synchronicity_slot_str = "N/A"

        convo_lengths = [c.message_count for c in self.processed_data.conversations if c.message_count > 0]
        if convo_lengths:
            self.processed_data.longest_conversation_messages = max(convo_lengths)
            self.processed_data.shortest_conversation_messages = min(convo_lengths)
            self.processed_data.average_conversation_messages = round(sum(convo_lengths) / len(convo_lengths), 1)
        logger.info(f"Conversation processing complete. Found {len(self.processed_data.conversations)} conversations.")

    def _process_feedback(self):
        feedback_json = self._load_json_file("message_feedback.json")
        if feedback_json:
            for fb_entry in feedback_json:
                rating = fb_entry.get('rating')
                if rating in ["thumbs_up", "thumbs_down"]:
                    self.processed_data.feedback_entries.append(Feedback(rating=rating))

            self.processed_data.feedback_total_insights_shared = len(self.processed_data.feedback_entries)
            self.processed_data.feedback_positive_count = sum(1 for fb in self.processed_data.feedback_entries if fb.rating == "thumbs_up")
            self.processed_data.feedback_improvement_suggestions_count = sum(1 for fb in self.processed_data.feedback_entries if fb.rating == "thumbs_down")
            logger.info(f"Feedback processing complete. Found {self.processed_data.feedback_total_insights_shared} feedback entries.")
        else:
            logger.info("No message_feedback.json found or it's empty.")


    def _calculate_derived_attributes(self):
        # Idea Galaxy (Word Cloud)
        all_user_text_for_cloud = " ".join(self.raw_user_messages_text).lower()
        words = re.findall(r'\b[a-z]{3,20}\b', all_user_text_for_cloud) # Words 3-20 chars
        filtered_words = [word for word in words if word not in STOP_WORDS]
        word_counts = Counter(filtered_words)
        self.processed_data.word_cloud_data = [{"text": word, "value": count} for word, count in word_counts.most_common(settings.WORD_CLOUD_TOP_N)]

        # Genesis Style
        analytical_hits = sum(1 for text in self.raw_user_messages_text if any(keyword in text.lower() for keyword in ANALYTICAL_KEYWORDS))
        creative_hits = sum(1 for text in self.raw_user_messages_text if any(keyword in text.lower() for keyword in CREATIVE_KEYWORDS))
        total_style_hits = analytical_hits + creative_hits
        if total_style_hits == 0:
            self.processed_data.genesis_style_analytical_pct = 50.0
            self.processed_data.genesis_style_creative_pct = 50.0
        else:
            self.processed_data.genesis_style_analytical_pct = round((analytical_hits / total_style_hits) * 100, 1)
            self.processed_data.genesis_style_creative_pct = round((creative_hits / total_style_hits) * 100, 1)

        # Badge: Rapid Responder
        # This flag is set in self.processed_data.attributes for consistency in how badges are flagged for the scoring/main proof
        self.processed_data.attributes = {} # Initialize if not already
        if self.processed_data.last_conversation_unix:
            last_data_date = datetime.fromtimestamp(self.processed_data.last_conversation_unix, tz=timezone.utc)
            seven_days_prior_to_last_data = last_data_date - timedelta(days=7)
            recent_interactions_count = 0
            for daily_data in self.processed_data.interaction_galaxy_data:
                interaction_date = datetime.strptime(daily_data["date"], '%Y-%m-%d').replace(tzinfo=timezone.utc)
                if interaction_date >= seven_days_prior_to_last_data:
                    recent_interactions_count += daily_data["count"]
            self.processed_data.attributes['badge_rapid_responder_flag'] = recent_interactions_count > settings.BADGE_RAPID_RESPONDER_INTERACTIONS_THRESHOLD
        else:
            self.processed_data.attributes['badge_rapid_responder_flag'] = False
        logger.info("Derived attributes calculated (word cloud, genesis style, badges).")


    def parse(self) -> ProcessedGPTData:
        logger.info("Starting full data parsing pipeline...")
        self._process_user_data()
        if not self.processed_data.user_profile:
            logger.error("User profile could not be processed. Aborting parsing.")
            # Consider raising an exception here to make it a hard stop for the PoC
            raise ValueError("User profile processing failed, essential for PoC.")

        self._process_conversations() # This populates core interaction metrics
        self._process_feedback()
        self._calculate_derived_attributes() # This populates word cloud, genesis, badge flags

        logger.info("Data parsing pipeline complete.")
        return self.processed_data