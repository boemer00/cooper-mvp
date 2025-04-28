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


def test_direct_url_valid_tiktok():
    """Test that a valid TikTok URL is accepted when provided directly."""
    finder = VideoFinder()
    valid_url = "https://www.tiktok.com/@username/video/1234567890123456789"
    result = finder.get_videos("any_topic", direct_url=valid_url)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == valid_url


def test_direct_url_invalid_tiktok():
    """Test that an invalid TikTok URL is rejected when provided directly."""
    finder = VideoFinder()
    invalid_urls = [
        "https://example.com/video/123",
        "https://facebook.com/watch/123",
        "not-a-url-at-all",
        "http://tiktok.fake.com/video/123",
    ]

    for url in invalid_urls:
        result = finder.get_videos("any_topic", direct_url=url)
        assert isinstance(result, list)
        assert len(result) == 0, f"URL '{url}' should be rejected"


def test_url_validation():
    """Test the URL validation method directly."""
    finder = VideoFinder()

    # Valid TikTok URLs
    valid_urls = [
        "https://www.tiktok.com/@username/video/1234567890123456789",
        "https://vm.tiktok.com/abcDEF123/",
        "http://www.tiktok.com/@user123_-./video/987654321",
    ]

    # Invalid TikTok URLs
    invalid_urls = [
        "https://example.com/video/123",
        "https://tiktok.com/video/123",  # Missing www or vm subdomain
        "ftp://www.tiktok.com/video/123",  # Wrong protocol
        "https://www.tiktok.com/video/123<script>",  # Invalid characters
    ]

    for url in valid_urls:
        assert finder.is_valid_tiktok_url(url), f"URL '{url}' should be valid"

    for url in invalid_urls:
        assert not finder.is_valid_tiktok_url(url), f"URL '{url}' should be invalid"
