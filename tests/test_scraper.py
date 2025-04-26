import pytest
from unittest.mock import patch, Mock, MagicMock
import time
from typing import Dict, Any, List

from src.scraper import Scraper, ScrapeConfig, VideoData, RequestError, ApifyClientError


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
        """Test that start_scrape returns a valid job ID using Apify client."""
        # Mock the task client and call method
        mock_task_client = Mock()
        mock_task_client.call.return_value = {"id": "test_job_id"}

        # Mock the client.task() method to return our mock task client
        mock_scraper._apify_client.task = Mock(return_value=mock_task_client)

        # Call the method to test
        job_id = mock_scraper.start_scrape(sample_config)

        # Assert job_id is a string and non-empty
        assert isinstance(job_id, str)
        assert job_id == "test_job_id"

        # Verify the task client was called with correct parameters
        mock_scraper._apify_client.task.assert_called_once_with(mock_scraper.actor_task_id)
        mock_task_client.call.assert_called_once()

        # Get the arguments passed to call()
        args, kwargs = mock_task_client.call.call_args
        assert "run_input" in kwargs
        assert kwargs["run_input"] == sample_config.model_dump(by_alias=True)

    def test_start_scrape_with_webhook(self, mock_scraper, sample_config):
        """Test start_scrape with webhook URL."""
        mock_scraper.webhook_url = "https://example.com/webhook"

        mock_response = Mock()
        mock_response.json.return_value = {"job_id": "webhook_job_id"}
        mock_response.raise_for_status.return_value = None

        with patch.object(mock_scraper._http_client, 'post', return_value=mock_response):
            job_id = mock_scraper.start_scrape(sample_config)

            assert job_id == "webhook_job_id"
            mock_scraper._http_client.post.assert_called_once_with(
                mock_scraper.webhook_url,
                json=sample_config.model_dump(by_alias=True)
            )

    def test_start_scrape_client_error(self, mock_scraper, sample_config):
        """Test that start_scrape raises ApifyClientError on client failure."""
        # Mock the task client to raise an exception
        mock_task_client = Mock()
        mock_task_client.call.side_effect = Exception("Client error")

        mock_scraper._apify_client.task = Mock(return_value=mock_task_client)

        # Check that ApifyClientError is raised
        with pytest.raises(ApifyClientError):
            mock_scraper.start_scrape(sample_config)

    def test_get_result_with_apify_client(self, mock_scraper, sample_response_data):
        """Test that get_result uses Apify client and returns proper data."""
        # Mock the run client
        mock_run_client = Mock()

        # Mock the run client's wait_for_finish method
        mock_run_client.wait_for_finish.return_value = {
            "id": "test_run_id",
            "status": "SUCCEEDED",
            "defaultDatasetId": "test_dataset_id"
        }

        # Mock the dataset client
        mock_dataset_client = Mock()
        mock_dataset_client.list_items.return_value = {
            "items": sample_response_data["items"]
        }

        # Set up the mock chain
        mock_scraper._apify_client.run = Mock(return_value=mock_run_client)
        mock_scraper._apify_client.dataset = Mock(return_value=mock_dataset_client)

        # Call the method to test
        results = mock_scraper.get_result("test_job_id")

        # Check the results
        assert len(results) == 2
        assert all(isinstance(item, VideoData) for item in results)
        assert results[0].url == "https://example.com/video1"
        assert results[1].url == "https://example.com/video2"

        # Verify correct methods were called
        mock_scraper._apify_client.run.assert_called_once_with("test_job_id")
        mock_run_client.wait_for_finish.assert_called_once_with(wait_secs=mock_scraper._timeout)
        mock_scraper._apify_client.dataset.assert_called_once_with("test_dataset_id")
        mock_dataset_client.list_items.assert_called_once()

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

        with patch.object(mock_scraper._http_client, 'get') as mock_get:
            mock_get.side_effect = status_responses

            results = mock_scraper.get_result("webhook_job_id")

            assert len(results) == 2
            assert results[0].url == "https://example.com/video1"
            assert results[1].url == "https://example.com/video2"
            assert mock_get.call_count == 2

            # Check first call was to correct webhook URL
            args, _ = mock_get.call_args_list[0]
            assert args[0] == "https://example.com/webhook/status/webhook_job_id"

    def test_get_result_timeout_with_apify_client(self, mock_scraper):
        """Test that get_result raises TimeoutError when client times out."""
        # Mock the run client
        mock_run_client = Mock()

        # Make wait_for_finish raise TimeoutError
        mock_run_client.wait_for_finish.side_effect = TimeoutError("Timed out")

        # Set up the mock
        mock_scraper._apify_client.run = Mock(return_value=mock_run_client)

        # Check that TimeoutError is raised
        with pytest.raises(TimeoutError):
            mock_scraper.get_result("test_job_id")

        # Verify correct methods were called
        mock_scraper._apify_client.run.assert_called_once_with("test_job_id")
        mock_run_client.wait_for_finish.assert_called_once_with(wait_secs=mock_scraper._timeout)

    def test_get_result_failed_run(self, mock_scraper):
        """Test that get_result raises ApifyClientError when run fails."""
        # Mock the run client
        mock_run_client = Mock()

        # Mock a failed run
        mock_run_client.wait_for_finish.return_value = {
            "id": "test_run_id",
            "status": "FAILED",
        }

        # Set up the mock
        mock_scraper._apify_client.run = Mock(return_value=mock_run_client)

        # Check that ApifyClientError is raised
        with pytest.raises(ApifyClientError):
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
