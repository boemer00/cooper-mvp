from typing import List, Dict


class VideoFinder:
    def __init__(self):
        """
        Initialize with a mapping from topic → List[str] of 10 URLs.
        """
        self._video_mapping: Dict[str, List[str]] = {
            "cooking": [
                "https://www.tiktok.com/cooking/video1",
                "https://www.tiktok.com/cooking/video2",
                "https://www.tiktok.com/cooking/video3",
                "https://www.tiktok.com/cooking/video4",
                "https://www.tiktok.com/cooking/video5",
                "https://www.tiktok.com/cooking/video6",
                "https://www.tiktok.com/cooking/video7",
                "https://www.tiktok.com/cooking/video8",
                "https://www.tiktok.com/cooking/video9",
                "https://www.tiktok.com/cooking/video10",
            ],
            "fitness": [
                "https://www.tiktok.com/fitness/video1",
                "https://www.tiktok.com/fitness/video2",
                "https://www.tiktok.com/fitness/video3",
                "https://www.tiktok.com/fitness/video4",
                "https://www.tiktok.com/fitness/video5",
                "https://www.tiktok.com/fitness/video6",
                "https://www.tiktok.com/fitness/video7",
                "https://www.tiktok.com/fitness/video8",
                "https://www.tiktok.com/fitness/video9",
                "https://www.tiktok.com/fitness/video10",
            ]
        }

    def get_videos(self, topic: str) -> List[str]:
        """
        Given a topic, return exactly 10 hardcoded TikTok URLs or [] if unknown.

        Args:
            topic: A string representing the topic of interest

        Returns:
            A list of 10 TikTok video URLs for known topics, or an empty list for unknown topics
        """
        return self._video_mapping.get(topic, [])
