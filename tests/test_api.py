from fastapi.testclient import TestClient
import pytest
import traceback
from unittest.mock import MagicMock, patch
from src.app import app
from src.scraper import VideoData, ScrapeConfig

client = TestClient(app)

# Debug middleware to capture responses and exceptions
@app.middleware("http")
async def debug_errors(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        print(f"Exception in middleware: {exc}")
        print(traceback.format_exc())
        raise

def test_chat_endpoint_success(monkeypatch):
    # Mock test data
    test_videos = ["https://example.com/video1", "https://example.com/video2"]
    test_video_data = [
        VideoData(
            url="https://example.com/video1",
            title="Test Video 1",
            transcript="Test transcript 1",
            audio_features={"pitch": 0.5, "tempo": 0.7},
            metadata={"likes": 100, "views": 1000},
            comments=["Great video!", "Nice content"]
        ),
        VideoData(
            url="https://example.com/video2",
            title="Test Video 2",
            transcript="Test transcript 2",
            audio_features={"pitch": 0.6, "tempo": 0.8},
            metadata={"likes": 200, "views": 2000},
            comments=["Awesome!", "Very helpful"]
        )
    ]
    test_text_emotions = {"joy": 0.5, "sadness": 0.2, "anger": 0.1}
    test_audio_emotions = {"joy": 0.3, "sadness": 0.4, "anger": 0.2}
    test_correlations = {"joy_vs_likes": 1.2, "sadness_vs_views": -0.8}
    test_insights = ["Insight 1", "Insight 2"]
    test_pr_hooks = ["PR Hook 1", "PR Hook 2"]

    # Mock VideoFinder
    mock_video_finder = MagicMock()
    mock_video_finder.get_videos.return_value = test_videos
    monkeypatch.setattr("src.app.VideoFinder", lambda: mock_video_finder)

    # Create a proper scraper mock that validates the config
    class MockScraper:
        def start_scrape(self, config):
            print("Mock scraper start_scrape called")
            # Verify config has required fields
            assert isinstance(config, ScrapeConfig)
            assert config.commentsPerPost == 10
            assert config.excludePinnedPosts is True
            assert config.maxRepliesPerComment == 5
            assert config.resultsPerPage == 20
            assert config.postURLs == test_videos
            return "mock_job_id"

        def get_result(self):
            print("Mock scraper get_result called")
            return test_video_data

    monkeypatch.setattr("src.app.Scraper", MockScraper)

    # Mock TextEmotionAnalyzer
    mock_text_analyzer = MagicMock()
    mock_text_analyzer.analyze.return_value = test_text_emotions
    monkeypatch.setattr("src.app.TextEmotionAnalyzer", lambda: mock_text_analyzer)

    # Mock AudioEmotionAnalyzer
    mock_audio_analyzer = MagicMock()
    mock_audio_analyzer.analyze.return_value = test_audio_emotions
    monkeypatch.setattr("src.app.AudioEmotionAnalyzer", lambda: mock_audio_analyzer)

    # Mock Correlator
    mock_correlator = MagicMock()
    mock_correlator.compute.return_value = test_correlations
    monkeypatch.setattr("src.app.Correlator", lambda: mock_correlator)

    # Mock InsightGenerator
    mock_insight_generator = MagicMock()
    mock_insight_generator.generate.return_value = test_insights
    mock_insight_generator.suggest_pr_hooks.return_value = test_pr_hooks
    monkeypatch.setattr("src.app.InsightGenerator", lambda: mock_insight_generator)

    # Make request
    response = client.get("/chat", params={"query": "cooking", "limit": 2})

    # Print debug info
    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Response body: {response.json()}")

    # Assertions
    assert response.status_code == 200
    json_response = response.json()

    assert json_response == {
        "videos": test_videos,
        "emotions": {
            "text": test_text_emotions,
            "audio": test_audio_emotions
        },
        "correlations": test_correlations,
        "insights": test_insights,
        "pr_hooks": test_pr_hooks
    }

    # Verify only the VideoFinder was called with expected parameters
    mock_video_finder.get_videos.assert_called_once_with("cooking", direct_url=None)

def test_chat_endpoint_direct_url(monkeypatch):
    # Test data for direct URL
    direct_tiktok_url = "https://www.tiktok.com/@username/video/1234567890"
    test_videos = [direct_tiktok_url]
    test_video_data = [
        VideoData(
            url=direct_tiktok_url,
            title="Direct TikTok Video",
            transcript="This is a direct TikTok video test",
            audio_features={"pitch": 0.6, "tempo": 0.7},
            metadata={"likes": 500, "views": 5000},
            comments=["Great!", "Love it"]
        )
    ]
    test_text_emotions = {"joy": 0.7, "sadness": 0.1}
    test_audio_emotions = {"joy": 0.6, "sadness": 0.2}
    test_correlations = {"joy_vs_likes": 0.8}
    test_insights = ["Direct URL Insight"]
    test_pr_hooks = ["Direct URL PR Hook"]

    # Mock VideoFinder
    mock_video_finder = MagicMock()
    mock_video_finder.get_videos.return_value = test_videos
    monkeypatch.setattr("src.app.VideoFinder", lambda: mock_video_finder)

    # Create scraper mock
    class MockScraper:
        def start_scrape(self, config):
            assert isinstance(config, ScrapeConfig)
            assert config.postURLs == test_videos
            return "mock_job_id"

        def get_result(self):
            return test_video_data

    monkeypatch.setattr("src.app.Scraper", MockScraper)

    # Mock other analyzers
    mock_text_analyzer = MagicMock()
    mock_text_analyzer.analyze.return_value = test_text_emotions
    monkeypatch.setattr("src.app.TextEmotionAnalyzer", lambda: mock_text_analyzer)

    mock_audio_analyzer = MagicMock()
    mock_audio_analyzer.analyze.return_value = test_audio_emotions
    monkeypatch.setattr("src.app.AudioEmotionAnalyzer", lambda: mock_audio_analyzer)

    mock_correlator = MagicMock()
    mock_correlator.compute.return_value = test_correlations
    monkeypatch.setattr("src.app.Correlator", lambda: mock_correlator)

    mock_insight_generator = MagicMock()
    mock_insight_generator.generate.return_value = test_insights
    mock_insight_generator.suggest_pr_hooks.return_value = test_pr_hooks
    monkeypatch.setattr("src.app.InsightGenerator", lambda: mock_insight_generator)

    # Make request with direct URL
    response = client.get("/chat", params={
        "query": "any_topic",
        "limit": 1,
        "url": direct_tiktok_url
    })

    # Assertions
    assert response.status_code == 200
    json_response = response.json()

    assert json_response == {
        "videos": test_videos,
        "emotions": {
            "text": test_text_emotions,
            "audio": test_audio_emotions
        },
        "correlations": test_correlations,
        "insights": test_insights,
        "pr_hooks": test_pr_hooks
    }

    # Verify VideoFinder was called with expected parameters
    mock_video_finder.get_videos.assert_called_once_with("any_topic", direct_url=direct_tiktok_url)

def test_chat_endpoint_invalid_direct_url(monkeypatch):
    # Mock VideoFinder to validate and reject invalid URL
    mock_video_finder = MagicMock()
    mock_video_finder.get_videos.return_value = []
    monkeypatch.setattr("src.app.VideoFinder", lambda: mock_video_finder)

    invalid_url = "https://example.com/not-a-tiktok-url"
    response = client.get("/chat", params={
        "query": "any_topic",
        "url": invalid_url
    })

    assert response.status_code == 404
    assert "No videos found" in response.json()["detail"]
    mock_video_finder.get_videos.assert_called_once_with("any_topic", direct_url=invalid_url)

def test_chat_endpoint_missing_query():
    response = client.get("/chat")
    assert response.status_code == 422

def test_chat_endpoint_no_videos_found(monkeypatch):
    # Mock VideoFinder to return empty list
    mock_video_finder = MagicMock()
    mock_video_finder.get_videos.return_value = []
    monkeypatch.setattr("src.app.VideoFinder", lambda: mock_video_finder)

    response = client.get("/chat", params={"query": "nonexistent_topic"})
    assert response.status_code == 404
    assert "No videos found" in response.json()["detail"]

def test_chat_endpoint_scraper_timeout(monkeypatch):
    # Mock VideoFinder
    mock_video_finder = MagicMock()
    mock_video_finder.get_videos.return_value = ["https://example.com/video1"]
    monkeypatch.setattr("src.app.VideoFinder", lambda: mock_video_finder)

    # Mock Scraper to raise TimeoutError
    mock_scraper = MagicMock()
    mock_scraper.start_scrape.side_effect = TimeoutError("Scraping timeout")
    monkeypatch.setattr("src.app.Scraper", lambda: mock_scraper)

    response = client.get("/chat", params={"query": "cooking"})
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]
