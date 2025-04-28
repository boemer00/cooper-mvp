from typing import List, Dict, Optional
import re


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

    def is_valid_tiktok_url(self, url: str) -> bool:
        """
        Validate if a URL is a proper TikTok video URL.

        Args:
            url: URL string to validate

        Returns:
            bool: True if valid TikTok URL, False otherwise
        """
        # Pattern for TikTok URLs (www.tiktok.com or vm.tiktok.com followed by path)
        pattern = r'^https?://(www|vm)\.tiktok\.com/[@a-zA-Z0-9_\-./]+$'
        return bool(re.match(pattern, url))

    def get_videos(self, topic: str, direct_url: Optional[str] = None) -> List[str]:
        """
        Given a topic, return exactly 10 hardcoded TikTok URLs or [] if unknown.
        If a direct URL is provided, validate and return it as a single-item list.

        Args:
            topic: A string representing the topic of interest
            direct_url: Optional direct TikTok URL to process

        Returns:
            A list of TikTok video URLs
        """
        if direct_url:
            if self.is_valid_tiktok_url(direct_url):
                return [direct_url]
            return []

        return self._video_mapping.get(topic, [])
