from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ValidationError
import time
import httpx
from apify_client import ApifyClient

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


class ApifyClientError(Exception):
    """Raised when an error occurs in the Apify client."""


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

        # Initialize both clients for flexibility
        self._http_client = httpx.Client(timeout=10)
        self._apify_client = ApifyClient(token=apify_token)

    def start_scrape(self, config: ScrapeConfig) -> str:
        """
        Trigger a scrape job and return a job_id/runId.

        :param config: ScrapeConfig containing scraping parameters
        :return: Job ID string for tracking the scrape job

        - If webhook_url is set: POST to webhook with config.json(by_alias=True).
        - Otherwise: Uses Apify client to start an Actor/Task run.
        """
        config_json = config.model_dump(by_alias=True)

        try:
            if self.webhook_url:
                # Use webhook URL if provided (keep existing functionality)
                response = self._http_client.post(self.webhook_url, json=config_json)
                response.raise_for_status()
                data = response.json()
                return data.get("job_id")
            else:
                # Use Apify client to start the task
                task_client = self._apify_client.task(self.actor_task_id)
                run = task_client.call(run_input=config_json)

                if not run:
                    raise ApifyClientError("Failed to start task, no run data returned")

                return run.get("id")
        except httpx.HTTPError as e:
            raise RequestError(f"Failed to start scrape job: {str(e)}") from e
        except Exception as e:
            raise ApifyClientError(f"Failed to start scrape job: {str(e)}") from e

    def get_result(self, job_id: str) -> List[VideoData]:
        """
        Wait for the job to complete and retrieve results.

        :param job_id: Job ID returned from start_scrape
        :return: List of VideoData objects containing scrape results

        Raises:
            TimeoutError: if not completed within self._timeout.
            RequestError: on repeated HTTP failures.
            ApifyClientError: on Apify client failures.
            ValidationError: if returned data doesn't match VideoData schema.
        """
        start_time = time.time()

        if self.webhook_url:
            # Use existing webhook polling logic
            while (time.time() - start_time) < self._timeout:
                try:
                    status_data = self._check_status(job_id)
                    status = status_data.get("status")

                    if status == "completed":
                        return self._parse_result(status_data)

                    # Wait before checking again
                    time.sleep(self._poll_interval)
                except httpx.HTTPError as e:
                    # Retry on HTTP errors
                    time.sleep(self._poll_interval)
                    continue
        else:
            # Use Apify client
            try:
                # Get the run client
                run_client = self._apify_client.run(job_id)

                # Wait for the run to finish with timeout
                run = run_client.wait_for_finish(wait_secs=self._timeout)

                if not run:
                    raise ApifyClientError(f"Failed to get run data for job {job_id}")

                if run.get("status") == "SUCCEEDED":
                    # Get dataset items
                    dataset_id = run.get("defaultDatasetId")
                    if not dataset_id:
                        raise ApifyClientError("No default dataset found in the run")

                    dataset_client = self._apify_client.dataset(dataset_id)
                    dataset_items = dataset_client.list_items().get("items", [])

                    return self._parse_result({"items": dataset_items})
                else:
                    # Run did not succeed
                    status = run.get("status", "UNKNOWN")
                    raise ApifyClientError(f"Run failed with status: {status}")
            except TimeoutError:
                # Re-raise timeout error
                raise
            except Exception as e:
                # Handle other exceptions
                raise ApifyClientError(f"Error retrieving run results: {str(e)}") from e

        # If we get here, we've timed out
        raise TimeoutError(f"Scrape job {job_id} did not complete within {self._timeout} seconds")

    def _check_status(self, job_id: str) -> Dict[str, Any]:
        """
        Helper to fetch job status JSON from webhook or Apify API.

        :param job_id: Job ID to check status for
        :return: Raw JSON response as dictionary

        Raises:
            RequestError: On HTTP request failures
            ApifyClientError: On Apify client failures
        """
        try:
            if self.webhook_url:
                # Check status via webhook (keep existing functionality)
                status_url = f"{self.webhook_url}/status/{job_id}"
                response = self._http_client.get(status_url)
                response.raise_for_status()
                return response.json()
            else:
                # Use Apify client to check run status
                run_client = self._apify_client.run(job_id)
                run_info = run_client.get()

                if not run_info:
                    raise ApifyClientError(f"Failed to get status for job {job_id}")

                return {"data": run_info}
        except httpx.HTTPError as e:
            raise RequestError(f"Failed to check job status: {str(e)}") from e
        except Exception as e:
            raise ApifyClientError(f"Failed to check job status: {str(e)}") from e

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
