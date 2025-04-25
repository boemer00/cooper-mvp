from typing import List, Dict, Any, Optional
import time
from pydantic import BaseModel, Field, ValidationError
import httpx

# --- Models ---

class ScrapeConfig(BaseModel):
    commentsPerPost: int
    excludePinnedPosts: bool
    maxRepliesPerComment: int
    resultsPerPage: int
    postURLs: List[str]


class VideoData(BaseModel):
    url: str
    comments: List[str]
    metadata: Dict[str, Any]


# --- Exceptions ---

class RequestError(Exception):
    """Raised when an HTTP request fails after retries."""


# --- Scraper ---

class Scraper:
    def __init__(
        self,
        apify_token: str,
        actor_task_id: str,
        webhook_url: Optional[str] = None,
        poll_interval: int = 5,
        timeout: int = 300,
    ):
        """
        Initialize the scraper with API credentials and configuration.

        :param apify_token: Your Apify API token.
        :param actor_task_id: The ID of your Apify actor/task.
        :param webhook_url: Optional n8n webhook to trigger instead of direct API.
        :param poll_interval: Seconds between status checks.
        :param timeout: Max seconds to wait for job completion.
        """
        self.apify_token = apify_token
        self.actor_task_id = actor_task_id
        self.webhook_url = webhook_url
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._client = httpx.Client(timeout=10)

    def start_scrape(self, config: ScrapeConfig) -> str:
        """
        Trigger a scrape job and return a job_id/runId.

        :param config: ScrapeConfig containing scraping parameters
        :return: Job ID string for tracking the scrape job

        - If webhook_url is set: POST to webhook with config.json(by_alias=True).
        - Otherwise: POST to Apify actor endpoint:
            https://api.apify.com/v2/actor-tasks/{actor_task_id}/runs?token={apify_token}
        """
        config_json = config.model_dump(by_alias=True)

        try:
            if self.webhook_url:
                # Use webhook URL if provided
                response = self._client.post(self.webhook_url, json=config_json)
                response.raise_for_status()
                data = response.json()
                return data.get("job_id")
            else:
                # Use Apify API directly
                url = f"https://api.apify.com/v2/actor-tasks/{self.actor_task_id}/runs?token={self.apify_token}"
                response = self._client.post(url, json=config_json)
                response.raise_for_status()
                data = response.json()
                return data.get("data", {}).get("id")
        except httpx.HTTPError as e:
            raise RequestError(f"Failed to start scrape job: {str(e)}") from e

    def get_result(self, job_id: str) -> List[VideoData]:
        """
        Poll until the job completes or timeout expires.

        :param job_id: Job ID returned from start_scrape
        :return: List of VideoData objects containing scrape results

        Raises:
            TimeoutError: if not completed within self._timeout.
            RequestError: on repeated HTTP failures.
            ValidationError: if returned data doesn't match VideoData schema.
        """
        start_time = time.time()

        while (time.time() - start_time) < self._timeout:
            try:
                status_data = self._check_status(job_id)

                # Handle webhook response
                if self.webhook_url:
                    status = status_data.get("status")
                    if status == "completed":
                        return self._parse_result(status_data)
                # Handle Apify response
                else:
                    status = status_data.get("data", {}).get("status")
                    if status == "SUCCEEDED":
                        # Fetch the output items from the default dataset
                        run_id = status_data.get("data", {}).get("id")
                        dataset_url = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={self.apify_token}"
                        response = self._client.get(dataset_url)
                        response.raise_for_status()
                        return self._parse_result({"items": response.json()})

                # Wait before checking again
                time.sleep(self._poll_interval)
            except httpx.HTTPError as e:
                # Retry on HTTP errors
                time.sleep(self._poll_interval)
                continue

        # If we get here, we've timed out
        raise TimeoutError(f"Scrape job {job_id} did not complete within {self._timeout} seconds")

    def _check_status(self, job_id: str) -> Dict[str, Any]:
        """
        Helper to fetch job status JSON from webhook or Apify API.

        :param job_id: Job ID to check status for
        :return: Raw JSON response as dictionary

        Raises:
            RequestError: On HTTP request failures
        """
        try:
            if self.webhook_url:
                # Check status via webhook
                status_url = f"{self.webhook_url}/status/{job_id}"
                response = self._client.get(status_url)
            else:
                # Check status via Apify API
                status_url = f"https://api.apify.com/v2/actor-runs/{job_id}?token={self.apify_token}"
                response = self._client.get(status_url)

            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise RequestError(f"Failed to check job status: {str(e)}") from e

    def _parse_result(self, raw_data: Dict[str, Any]) -> List[VideoData]:
        """
        Parse the actor's output payload into a list of VideoData.

        :param raw_data: Raw JSON response data
        :return: List of validated VideoData objects

        Raises:
            ValidationError: If data doesn't match expected schema
        """
        try:
            items = raw_data.get("items", [])
            return [VideoData(**item) for item in items]
        except (KeyError, TypeError) as e:
            raise ValidationError(f"Invalid data structure: {e}") from e
        except Exception as e:
            raise ValidationError(f"Failed to parse result: {e}") from e
