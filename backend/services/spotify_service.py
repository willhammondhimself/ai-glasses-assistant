"""Spotify API service for music control."""
import os
import logging
import base64
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_URL = "https://api.spotify.com/v1"

SCOPES = [
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "playlist-read-private",
    "user-library-read",
]


@dataclass
class SpotifyTrack:
    """Represents a Spotify track."""
    id: str
    name: str
    artist: str
    album: str
    duration_ms: int
    uri: str

    def to_voice_summary(self) -> str:
        return f"{self.name} by {self.artist}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "artist": self.artist,
            "album": self.album,
            "duration_ms": self.duration_ms,
            "uri": self.uri
        }


@dataclass
class PlaybackState:
    """Current playback state."""
    is_playing: bool
    track: Optional[SpotifyTrack]
    progress_ms: int
    device_name: str
    volume_percent: int
    shuffle: bool
    repeat: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_playing": self.is_playing,
            "track": self.track.to_dict() if self.track else None,
            "progress_ms": self.progress_ms,
            "device_name": self.device_name,
            "volume_percent": self.volume_percent,
            "shuffle": self.shuffle,
            "repeat": self.repeat
        }


class SpotifyService:
    """Spotify API wrapper for voice-controlled music."""

    def __init__(self):
        """Initialize Spotify service."""
        self.client_id = os.environ.get("SPOTIFY_CLIENT_ID")
        self.client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
        self.redirect_uri = os.environ.get("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

        # Load saved tokens
        self._load_tokens()

    def is_configured(self) -> bool:
        """Check if Spotify API credentials are set."""
        return bool(self.client_id and self.client_secret)

    def is_authenticated(self) -> bool:
        """Check if we have valid tokens."""
        return bool(self._access_token and self._token_expires and datetime.now() < self._token_expires)

    def get_auth_url(self) -> str:
        """Get OAuth authorization URL.

        Returns:
            URL to redirect user for Spotify auth
        """
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(SCOPES),
            "show_dialog": "true"
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{SPOTIFY_AUTH_URL}?{query}"

    async def authenticate_with_code(self, code: str) -> bool:
        """Exchange auth code for tokens.

        Args:
            code: Authorization code from Spotify callback

        Returns:
            True if successful
        """
        auth_header = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                SPOTIFY_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri
                }
            )

            if response.status_code != 200:
                logger.error(f"Spotify auth failed: {response.text}")
                return False

            data = response.json()
            self._access_token = data["access_token"]
            self._refresh_token = data.get("refresh_token")
            self._token_expires = datetime.now() + timedelta(seconds=data["expires_in"] - 60)

            self._save_tokens()
            logger.info("Spotify authenticated successfully")
            return True

    async def _refresh_access_token(self) -> bool:
        """Refresh the access token."""
        if not self._refresh_token:
            return False

        auth_header = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                SPOTIFY_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token
                }
            )

            if response.status_code != 200:
                logger.error(f"Token refresh failed: {response.text}")
                return False

            data = response.json()
            self._access_token = data["access_token"]
            self._token_expires = datetime.now() + timedelta(seconds=data["expires_in"] - 60)

            if "refresh_token" in data:
                self._refresh_token = data["refresh_token"]

            self._save_tokens()
            return True

    async def _ensure_token(self) -> bool:
        """Ensure we have a valid access token."""
        if not self._access_token:
            return False

        if self._token_expires and datetime.now() >= self._token_expires:
            return await self._refresh_access_token()

        return True

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make an API request to Spotify.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body
            params: Query parameters

        Returns:
            Response JSON or None
        """
        if not await self._ensure_token():
            raise RuntimeError("Not authenticated with Spotify")

        url = f"{SPOTIFY_API_URL}{endpoint}"
        headers = {"Authorization": f"Bearer {self._access_token}"}

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=data,
                params=params
            )

            if response.status_code == 204:
                return {}

            if response.status_code == 401:
                # Token expired, try refresh
                if await self._refresh_access_token():
                    headers["Authorization"] = f"Bearer {self._access_token}"
                    response = await client.request(
                        method, url, headers=headers, json=data, params=params
                    )

            if response.status_code >= 400:
                logger.error(f"Spotify API error: {response.status_code} - {response.text}")
                return None

            return response.json() if response.text else {}

    def _load_tokens(self):
        """Load saved tokens from file."""
        token_path = os.path.expanduser("~/.wham/spotify_token.json")
        if os.path.exists(token_path):
            import json
            try:
                with open(token_path) as f:
                    data = json.load(f)
                    self._access_token = data.get("access_token")
                    self._refresh_token = data.get("refresh_token")
                    if data.get("expires_at"):
                        self._token_expires = datetime.fromisoformat(data["expires_at"])
            except Exception as e:
                logger.warning(f"Failed to load Spotify tokens: {e}")

    def _save_tokens(self):
        """Save tokens to file."""
        import json
        token_path = os.path.expanduser("~/.wham/spotify_token.json")
        os.makedirs(os.path.dirname(token_path), exist_ok=True)

        with open(token_path, 'w') as f:
            json.dump({
                "access_token": self._access_token,
                "refresh_token": self._refresh_token,
                "expires_at": self._token_expires.isoformat() if self._token_expires else None
            }, f)

    async def get_current_playback(self) -> Optional[PlaybackState]:
        """Get current playback state.

        Returns:
            PlaybackState or None if nothing playing
        """
        data = await self._api_request("GET", "/me/player")

        if not data or not data.get("item"):
            return None

        track_data = data["item"]
        track = SpotifyTrack(
            id=track_data["id"],
            name=track_data["name"],
            artist=", ".join(a["name"] for a in track_data["artists"]),
            album=track_data["album"]["name"],
            duration_ms=track_data["duration_ms"],
            uri=track_data["uri"]
        )

        return PlaybackState(
            is_playing=data.get("is_playing", False),
            track=track,
            progress_ms=data.get("progress_ms", 0),
            device_name=data.get("device", {}).get("name", "Unknown"),
            volume_percent=data.get("device", {}).get("volume_percent", 50),
            shuffle=data.get("shuffle_state", False),
            repeat=data.get("repeat_state", "off")
        )

    async def play(self, uri: Optional[str] = None, context_uri: Optional[str] = None) -> bool:
        """Start or resume playback.

        Args:
            uri: Specific track URI to play
            context_uri: Playlist/album URI to play

        Returns:
            True if successful
        """
        data = {}
        if uri:
            data["uris"] = [uri]
        elif context_uri:
            data["context_uri"] = context_uri

        result = await self._api_request("PUT", "/me/player/play", data if data else None)
        return result is not None

    async def pause(self) -> bool:
        """Pause playback.

        Returns:
            True if successful
        """
        result = await self._api_request("PUT", "/me/player/pause")
        return result is not None

    async def skip_next(self) -> bool:
        """Skip to next track.

        Returns:
            True if successful
        """
        result = await self._api_request("POST", "/me/player/next")
        return result is not None

    async def skip_previous(self) -> bool:
        """Skip to previous track.

        Returns:
            True if successful
        """
        result = await self._api_request("POST", "/me/player/previous")
        return result is not None

    async def set_volume(self, volume: int) -> bool:
        """Set volume level.

        Args:
            volume: Volume percentage (0-100)

        Returns:
            True if successful
        """
        volume = max(0, min(100, volume))
        result = await self._api_request(
            "PUT",
            "/me/player/volume",
            params={"volume_percent": volume}
        )
        return result is not None

    async def search(self, query: str, search_type: str = "track", limit: int = 5) -> List[SpotifyTrack]:
        """Search for tracks/albums/artists.

        Args:
            query: Search query
            search_type: Type to search (track, album, artist, playlist)
            limit: Max results

        Returns:
            List of matching tracks
        """
        result = await self._api_request(
            "GET",
            "/search",
            params={
                "q": query,
                "type": search_type,
                "limit": limit
            }
        )

        if not result:
            return []

        tracks = []
        if "tracks" in result:
            for item in result["tracks"]["items"]:
                tracks.append(SpotifyTrack(
                    id=item["id"],
                    name=item["name"],
                    artist=", ".join(a["name"] for a in item["artists"]),
                    album=item["album"]["name"],
                    duration_ms=item["duration_ms"],
                    uri=item["uri"]
                ))

        return tracks

    async def play_search(self, query: str) -> Optional[SpotifyTrack]:
        """Search and play the first matching track.

        Args:
            query: Search query (song name, artist, etc.)

        Returns:
            Track that was played, or None
        """
        tracks = await self.search(query, limit=1)
        if not tracks:
            return None

        track = tracks[0]
        await self.play(uri=track.uri)
        return track

    async def get_user_playlists(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's playlists.

        Args:
            limit: Max playlists to return

        Returns:
            List of playlist info
        """
        result = await self._api_request(
            "GET",
            "/me/playlists",
            params={"limit": limit}
        )

        if not result:
            return []

        return [
            {
                "id": p["id"],
                "name": p["name"],
                "uri": p["uri"],
                "tracks_count": p["tracks"]["total"]
            }
            for p in result.get("items", [])
        ]

    async def play_playlist(self, playlist_name: str) -> Optional[str]:
        """Find and play a playlist by name.

        Args:
            playlist_name: Name of playlist to play

        Returns:
            Playlist name if found and playing, None otherwise
        """
        playlists = await self.get_user_playlists(limit=50)

        # Find best match
        playlist_lower = playlist_name.lower()
        for p in playlists:
            if playlist_lower in p["name"].lower():
                await self.play(context_uri=p["uri"])
                return p["name"]

        # Try searching for public playlists
        result = await self._api_request(
            "GET",
            "/search",
            params={
                "q": playlist_name,
                "type": "playlist",
                "limit": 1
            }
        )

        if result and result.get("playlists", {}).get("items"):
            playlist = result["playlists"]["items"][0]
            await self.play(context_uri=playlist["uri"])
            return playlist["name"]

        return None


# Global instance
_spotify_service: Optional[SpotifyService] = None


def get_spotify_service() -> SpotifyService:
    """Get or create global Spotify service instance."""
    global _spotify_service
    if _spotify_service is None:
        _spotify_service = SpotifyService()
    return _spotify_service
