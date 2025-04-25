from typing import List, Dict, Any
import statistics
from src.scraper import VideoData

class Correlator:
    def __init__(self):
        """
        Initialize correlation parameters.
        """
        self.emotions = [
            "joy", "sadness", "anger", "fear",
            "surprise", "disgust", "neutral"
        ]
        self.metadata_fields = [
            "likes", "comments", "shares", "views"
        ]

    def compute(
        self,
        video_data: List[VideoData],
        text_scores: Dict[str, float],
        audio_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Given scraped VideoData and emotion-score dicts,
        returns a mapping like {'happy_vs_hearts': 2.0, 'sad_vs_comments': 0.5, ...}.

        Args:
            video_data: List of VideoData objects containing metadata
            text_scores: Dictionary of emotion scores from text analysis
            audio_scores: Dictionary of emotion scores from audio analysis

        Returns:
            Dictionary of correlation metrics between emotions and metadata
        """
        results = {}

        # Merge emotion scores (simple average)
        combined_scores = {}
        for emotion in self.emotions:
            text_value = text_scores.get(emotion, 0.0)
            audio_value = audio_scores.get(emotion, 0.0)
            combined_scores[emotion] = (text_value + audio_value) / 2

        # Extract metadata averages from videos
        metadata_values = {field: 0.0 for field in self.metadata_fields}
        video_count = len(video_data)

        if video_count == 0:
            return {}

        # Calculate average values for each metadata field
        for video in video_data:
            for field in self.metadata_fields:
                if field in video.metadata:
                    metadata_values[field] += float(video.metadata.get(field, 0))

        # Calculate averages
        for field in metadata_values:
            metadata_values[field] /= video_count

        # Calculate correlations between emotions and metadata
        for emotion in combined_scores:
            for field in metadata_values:
                if metadata_values[field] > 0:
                    # Simple ratio calculation: emotion_score / metadata_value
                    # Higher ratio = stronger correlation
                    correlation = combined_scores[emotion] / metadata_values[field]
                    results[f"{emotion}_vs_{field}"] = round(correlation * 100, 2)
                else:
                    results[f"{emotion}_vs_{field}"] = 0.0

        # Add additional metrics: comment sentiment per emotion
        comment_count = sum(len(video.comments) for video in video_data)
        if comment_count > 0:
            for emotion in combined_scores:
                results[f"{emotion}_comment_ratio"] = round(
                    combined_scores[emotion] * 100 / comment_count, 2
                )

        return results
