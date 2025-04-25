import pytest
import sys
import os

# Add the parent directory to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_finder import VideoFinder


def test_known_topic_returns_10_urls():
    """Test that a known topic returns exactly 10 URLs."""
    finder = VideoFinder()

    # Test for the "cooking" topic
    result = finder.get_videos("cooking")
    assert isinstance(result, list)
    assert len(result) == 10
    assert all(isinstance(url, str) for url in result)

    # Test for the "fitness" topic
    result = finder.get_videos("fitness")
    assert isinstance(result, list)
    assert len(result) == 10
    assert all(isinstance(url, str) for url in result)


def test_unknown_topic_returns_empty_list():
    """Test that an unknown topic returns an empty list."""
    finder = VideoFinder()
    result = finder.get_videos("no_such_topic")
    assert isinstance(result, list)
    assert len(result) == 0
    assert result == []
