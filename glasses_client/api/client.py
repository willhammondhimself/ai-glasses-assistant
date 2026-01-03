"""
REST API Client

HTTP client for backend API communication.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API client configuration."""
    base_url: str = "http://localhost:8000"
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


class APIClient:
    """
    REST API client for the AI Glasses Coach backend.

    Features:
    - Async HTTP requests
    - Automatic retries with backoff
    - Connection pooling
    - Error handling
    """

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(
                base_url=self.config.base_url,
                timeout=timeout
            )
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a GET request.

        Args:
            path: API path (e.g., "/health")
            params: Query parameters

        Returns:
            Response JSON as dictionary
        """
        return await self._request("GET", path, params=params)

    async def post(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a POST request.

        Args:
            path: API path
            data: Request body

        Returns:
            Response JSON as dictionary
        """
        return await self._request("POST", path, json=data)

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retries.

        Args:
            method: HTTP method
            path: API path
            **kwargs: Additional aiohttp request arguments

        Returns:
            Response JSON

        Raises:
            APIError: On request failure
        """
        session = await self._get_session()
        last_error = None

        for attempt in range(self.config.retry_attempts):
            try:
                async with session.request(method, path, **kwargs) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise APIError(
                            f"API error {response.status}: {error_text}",
                            status=response.status
                        )

                    return await response.json()

            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")

                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(
                        self.config.retry_delay_seconds * (attempt + 1)
                    )

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                break

        raise APIError(f"Request failed after {self.config.retry_attempts} attempts: {last_error}")

    # Convenience methods for specific endpoints

    async def health_check(self) -> bool:
        """Check if the backend is healthy."""
        try:
            result = await self.get("/health")
            return result.get("status") == "ok"
        except Exception:
            return False

    async def solve_math(self, problem: str) -> Dict[str, Any]:
        """
        Solve a math problem and get AR-formatted response.

        Args:
            problem: Math problem text

        Returns:
            PaginatedResponse with slides
        """
        return await self.post("/ar/math/solve", {"problem": problem})

    async def get_router_status(self) -> Dict[str, Any]:
        """Get router status and budget info."""
        return await self.get("/router/status")


class APIError(Exception):
    """API request error."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status
