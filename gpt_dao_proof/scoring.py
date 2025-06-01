"""Scoring logic for processed ChatGPT data."""
from typing import Dict

from gpt_dao_proof.models.gpt_data import ProcessedGPTData
from gpt_dao_proof.config import settings

class GPTContributionScorer:
    def __init__(self):
        # This is the denominator used for normalizing raw points to a 0-1 score.
        self.points_denominator = float(settings.MAX_INFOGRAPHIC_POINTS)

    def calculate_score_and_points(self, data: ProcessedGPTData) -> tuple[float, Dict[str, int]]:
        """
        Calculates a normalized score (0-1) for the total current export and a detailed raw points breakdown.
        The normalized score is `total_achieved_raw_points / points_denominator`, clamped to 1.0.
        This represents the full value of the *current* export file.
        """
        points = {}
        total_achieved_raw_points = 0

        # Conversation Points - New Tiers
        points['conversations'] = 0
        if data.total_conversations_initiated > 1000: points['conversations'] = 90
        elif data.total_conversations_initiated > 500: points['conversations'] = 70
        elif data.total_conversations_initiated > 200: points['conversations'] = 50
        elif data.total_conversations_initiated > 50: points['conversations'] = 30
        elif data.total_conversations_initiated > 10: points['conversations'] = 15
        elif data.total_conversations_initiated > 0: points['conversations'] = 5
        total_achieved_raw_points += points['conversations']

        points['user_prompts'] = 0
        if data.total_user_prompts_penned > 2000: points['user_prompts'] = 50
        elif data.total_user_prompts_penned > 500: points['user_prompts'] = 30
        elif data.total_user_prompts_penned > 100: points['user_prompts'] = 15
        elif data.total_user_prompts_penned > 0: points['user_prompts'] = 5
        total_achieved_raw_points += points['user_prompts']

        points['activity_span'] = 0
        if data.total_days_active > 365: points['activity_span'] = 50
        elif data.total_days_active > 180: points['activity_span'] = 30
        elif data.total_days_active > 30: points['activity_span'] = 15
        elif data.total_days_active > 0: points['activity_span'] = 5
        total_achieved_raw_points += points['activity_span']

        points['model_quality'] = 0
        if data.total_assistant_model_messages_with_slug > 0:
            gpt4_usage_pct = data.gpt4_message_count / data.total_assistant_model_messages_with_slug
            if gpt4_usage_pct >= settings.BADGE_GPT4_USAGE_THRESHOLD_PCT: points['model_quality'] = 50
            elif gpt4_usage_pct > 0.2: points['model_quality'] = 30
            else: points['model_quality'] = 10
        total_achieved_raw_points += points['model_quality']

        points['conversation_depth'] = 0
        if data.average_conversation_messages > 10: points['conversation_depth'] = 30
        elif data.average_conversation_messages >= 5: points['conversation_depth'] = 15
        elif data.average_conversation_messages > 0 : points['conversation_depth'] = 5
        total_achieved_raw_points += points['conversation_depth']

        points['code_usage'] = 0
        if data.code_weaver_prompts_count > 20: points['code_usage'] = 30
        elif data.code_weaver_prompts_count > 5: points['code_usage'] = 15
        elif data.code_weaver_prompts_count > 0: points['code_usage'] = 5
        total_achieved_raw_points += points['code_usage']

        points['feedback_engagement'] = 0
        if data.feedback_total_insights_shared >= settings.BADGE_FEEDBACK_GOLD_COUNT: points['feedback_engagement'] = 40
        elif data.feedback_total_insights_shared >= settings.BADGE_FEEDBACK_SILVER_COUNT: points['feedback_engagement'] = 25
        elif data.feedback_total_insights_shared >= settings.BADGE_FEEDBACK_BRONZE_COUNT: points['feedback_engagement'] = 10
        elif data.feedback_total_insights_shared > 0: points['feedback_engagement'] = 5
        total_achieved_raw_points += points['feedback_engagement']

        if data.user_profile and data.user_profile.is_plus_user:
            points['plus_user_bonus'] = 20
            total_achieved_raw_points += points['plus_user_bonus']
        else:
            points['plus_user_bonus'] = 0

        max_achievable_raw_points_from_categories = 90 + 50 + 50 + 50 + 30 + 30 + 40 + 20 # Sum of max points from each category = 360

        normalized_score_of_current_export = 0.0
        if self.points_denominator > 0: # Denominator is settings.MAX_INFOGRAPHIC_POINTS (e.g., 1000)
            normalized_score_of_current_export = min(1.0, total_achieved_raw_points / self.points_denominator)

        points['total_achieved_raw_points'] = total_achieved_raw_points
        points['max_possible_raw_points_from_categories'] = max_achievable_raw_points_from_categories

        return round(normalized_score_of_current_export, 6), points