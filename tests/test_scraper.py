import pytest
from unittest.mock import patch, Mock, MagicMock
import time
from typing import Dict, Any, List

from src.scraper import Scraper, ScrapeConfig, VideoData, RequestError


@pytest.fixture
def mock_scraper():
    """Create a Scraper instance with mock configuration."""
    return Scraper(
        apify_token="test_token",
        actor_task_id="test_task_id",
        poll_interval=1,  # Fast polling for tests
        timeout=5,        # Short timeout for tests
    )


@pytest.fixture
def sample_config():
    """Create a sample ScrapeConfig for testing."""
    return ScrapeConfig(
        commentsPerPost=10,
        excludePinnedPosts=True,
        maxRepliesPerComment=5,
        resultsPerPage=20,
        postURLs=["https://example.com/post1", "https://example.com/post2"]
    )


@pytest.fixture
def sample_response_data():
    """Sample data that would be returned from the API."""
    return {
        "items": [
            {
                "url": "https://example.com/video1",
                "comments": ["Great video!", "Nice content"],
                "metadata": {"duration": 120, "title": "Test Video 1"}
            },
            {
                "url": "https://example.com/video2",
                "comments": ["Awesome!", "Very informative"],
                "metadata": {"duration": 180, "title": "Test Video 2"}
            }
        ]
    }


class TestScraper:
    """Tests for the Scraper class."""

    def test_start_scrape_returns_job_id(self, mock_scraper, sample_config):
        """Test that start_scrape returns a valid job ID."""
        # Mock the httpx client post method
        mock_response = Mock()
        mock_response.json.return_value = {"data": {"id": "test_job_id"}}
        mock_response.raise_for_status.return_value = None

        with patch.object(mock_scraper._client, 'post', return_value=mock_response):
            job_id = mock_scraper.start_scrape(sample_config)

            # Assert job_id is a string and non-empty
            assert isinstance(job_id, str)
            assert job_id == "test_job_id"

            # Verify the post was called with correct URL and data
            expected_url = f"https://api.apify.com/v2/actor-tasks/{mock_scraper.actor_task_id}/runs?token={mock_scraper.apify_token}"
            mock_scraper._client.post.assert_called_once()
            args, kwargs = mock_scraper._client.post.call_args
            assert args[0] == expected_url
            assert "json" in kwargs

    def test_start_scrape_with_webhook(self, mock_scraper, sample_config):
        """Test start_scrape with webhook URL."""
        mock_scraper.webhook_url = "https://example.com/webhook"

        mock_response = Mock()
        mock_response.json.return_value = {"job_id": "webhook_job_id"}
        mock_response.raise_for_status.return_value = None

        with patch.object(mock_scraper._client, 'post', return_value=mock_response):
            job_id = mock_scraper.start_scrape(sample_config)

            assert job_id == "webhook_job_id"
            mock_scraper._client.post.assert_called_once_with(
                mock_scraper.webhook_url,
                json=sample_config.model_dump(by_alias=True)
            )

    def test_get_result_polls_until_success(self, mock_scraper, sample_response_data):
        """Test that get_result polls until job succeeds and returns proper data."""
        # Create a sequence of mock responses for status checks
        status_responses = [
            # First call - job is running
            Mock(
                json=Mock(return_value={"data": {"status": "RUNNING", "id": "test_job_id"}}),
                raise_for_status=Mock(return_value=None)
            ),
            # Second call - job is still running
            Mock(
                json=Mock(return_value={"data": {"status": "RUNNING", "id": "test_job_id"}}),
                raise_for_status=Mock(return_value=None)
            ),
            # Third call - job succeeded
            Mock(
                json=Mock(return_value={"data": {"status": "SUCCEEDED", "id": "test_job_id"}}),
                raise_for_status=Mock(return_value=None)
            )
        ]

        # Mock for the dataset items response
        items_response = Mock(
            json=Mock(return_value=sample_response_data["items"]),
            raise_for_status=Mock(return_value=None)
        )

        with patch.object(mock_scraper._client, 'get') as mock_get:
            # Set up the mock to return different responses on subsequent calls
            mock_get.side_effect = status_responses + [items_response]

            # Call the method to test
            results = mock_scraper.get_result("test_job_id")

            # Check the results
            assert len(results) == 2
            assert all(isinstance(item, VideoData) for item in results)
            assert results[0].url == "https://example.com/video1"
            assert results[1].url == "https://example.com/video2"

            # Verify correct number of calls (3 status checks + 1 items fetch)
            assert mock_get.call_count == 4

    def test_get_result_with_webhook(self, mock_scraper, sample_response_data):
        """Test get_result with webhook response."""
        mock_scraper.webhook_url = "https://example.com/webhook"

        # Mock responses for webhook status checks
        status_responses = [
            # First call - job is running
            Mock(
                json=Mock(return_value={"status": "running"}),
                raise_for_status=Mock(return_value=None)
            ),
            # Second call - job completed
            Mock(
                json=Mock(return_value={"status": "completed", "items": sample_response_data["items"]}),
                raise_for_status=Mock(return_value=None)
            )
        ]

        with patch.object(mock_scraper._client, 'get') as mock_get:
            mock_get.side_effect = status_responses

            results = mock_scraper.get_result("webhook_job_id")

            assert len(results) == 2
            assert results[0].url == "https://example.com/video1"
            assert results[1].url == "https://example.com/video2"
            assert mock_get.call_count == 2

            # Check first call was to correct webhook URL
            args, _ = mock_get.call_args_list[0]
            assert args[0] == "https://example.com/webhook/status/webhook_job_id"

    def test_get_result_timeout(self, mock_scraper):
        """Test that get_result raises TimeoutError when polling exceeds timeout."""
        # Mock response that always returns "RUNNING" status
        mock_response = Mock(
            json=Mock(return_value={"data": {"status": "RUNNING"}}),
            raise_for_status=Mock(return_value=None)
        )

        with patch.object(mock_scraper._client, 'get', return_value=mock_response):
            # Set a very short timeout to speed up the test
            mock_scraper._timeout = 2

            # Check that TimeoutError is raised
            with pytest.raises(TimeoutError):
                mock_scraper.get_result("test_job_id")

    def test_parse_result_validation_error(self, mock_scraper):
        """Test that _parse_result raises ValidationError on invalid data."""
        invalid_data = {
            "items": [
                {
                    # Missing required fields
                    "url": "https://example.com/video1"
                }
            ]
        }

        with pytest.raises(Exception):  # Could be ValidationError or similar
            mock_scraper._parse_result(invalid_data)
