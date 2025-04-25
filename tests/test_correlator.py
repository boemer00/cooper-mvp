import pytest
from src.correlator import Correlator
from src.scraper import VideoData

def test_correlator_init():
    """Test that Correlator initializes correctly."""
    corr = Correlator()
    assert hasattr(corr, "emotions")
    assert hasattr(corr, "metadata_fields")

def test_compute_returns_expected_keys():
    """Test that compute returns a dictionary with expected keys."""
    corr = Correlator()

    # Create dummy VideoData
    video_data = [
        VideoData(
            url="https://example.com/video1",
            comments=["Great video!", "Love it!"],
            metadata={"likes": 100, "comments": 2, "shares": 50, "views": 1000}
        ),
        VideoData(
            url="https://example.com/video2",
            comments=["Nice content", "Could be better"],
            metadata={"likes": 80, "comments": 2, "shares": 30, "views": 800}
        )
    ]

    # Create dummy emotion scores
    text_scores = {
        "joy": 0.6,
        "sadness": 0.1,
        "anger": 0.05,
        "fear": 0.05,
        "surprise": 0.1,
        "disgust": 0.0,
        "neutral": 0.1
    }

    audio_scores = {
        "joy": 0.7,
        "sadness": 0.05,
        "anger": 0.05,
        "fear": 0.0,
        "surprise": 0.1,
        "disgust": 0.0,
        "neutral": 0.1
    }

    # Call compute
    result = corr.compute(video_data, text_scores, audio_scores)

    # Check that result is a dictionary
    assert isinstance(result, dict)

    # Check that we have the expected keys
    for emotion in corr.emotions:
        for field in corr.metadata_fields:
            assert f"{emotion}_vs_{field}" in result
            assert isinstance(result[f"{emotion}_vs_{field}"], float)

    # Check for comment ratio metrics
    for emotion in corr.emotions:
        assert f"{emotion}_comment_ratio" in result
        assert isinstance(result[f"{emotion}_comment_ratio"], float)

def test_compute_handles_empty_input():
    """Test that compute handles empty input gracefully."""
    corr = Correlator()

    # Call compute with empty video data
    result = corr.compute([], {}, {})

    # Check that result is an empty dictionary
    assert result == {}

def test_compute_handles_missing_metadata():
    """Test that compute handles missing metadata fields gracefully."""
    corr = Correlator()

    # Create VideoData with missing metadata fields
    video_data = [
        VideoData(
            url="https://example.com/video1",
            comments=["Test comment"],
            metadata={"likes": 100}  # Missing other fields
        )
    ]

    # Create dummy emotion scores
    text_scores = {"joy": 0.5}
    audio_scores = {"joy": 0.5}

    # Call compute
    result = corr.compute(video_data, text_scores, audio_scores)

    # Check that we have expected keys
    assert "joy_vs_likes" in result
    assert isinstance(result["joy_vs_likes"], float)

    # Fields with zero values should result in zero correlation
    assert "joy_vs_comments" in result
    assert result["joy_vs_comments"] == 0.0
