"""Music voice tool for Spotify control."""
import re
import logging
from typing import Optional
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class MusicVoiceTool(VoiceTool):
    """Voice-controlled music via Spotify."""

    name = "music"
    description = "Control music playback via Spotify"

    keywords = [
        r"\bplay\s+(?!poker)",  # "play" but not "play poker"
        r"\bpause\b",
        r"\bstop\s+(?:the\s+)?music\b",
        r"\bskip\b",
        r"\bnext\s+(?:song|track)\b",
        r"\bprevious\s+(?:song|track)\b",
        r"\bwhat('s| is)\s+playing\b",
        r"\bwhat\s+song\b",
        r"\bvolume\b",
        r"\bspotify\b",
        r"\bmusic\b",
        r"\bsong\b",
        r"\bplaylist\b",
    ]

    priority = 9  # High priority for media control

    def __init__(self):
        self._spotify = None

    def _get_spotify(self):
        """Get Spotify service (lazy load)."""
        if self._spotify is None:
            from backend.services.spotify_service import get_spotify_service
            self._spotify = get_spotify_service()
        return self._spotify

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute music command.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with music info
        """
        query_lower = query.lower()
        spotify = self._get_spotify()

        # Check if configured
        if not spotify.is_configured():
            return VoiceToolResult(
                success=False,
                message="Spotify is not configured. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.",
                data={"needs_setup": True}
            )

        # Check if authenticated
        if not spotify.is_authenticated():
            auth_url = spotify.get_auth_url()
            return VoiceToolResult(
                success=False,
                message="Please authenticate with Spotify first.",
                data={"needs_auth": True, "auth_url": auth_url}
            )

        try:
            # Route to appropriate handler
            if self._is_pause_request(query_lower):
                return await self._handle_pause()

            if self._is_skip_next(query_lower):
                return await self._handle_skip_next()

            if self._is_skip_previous(query_lower):
                return await self._handle_skip_previous()

            if self._is_whats_playing(query_lower):
                return await self._handle_whats_playing()

            if self._is_volume_request(query_lower):
                return await self._handle_volume(query_lower)

            if self._is_play_request(query_lower):
                return await self._handle_play(query)

            # Default: check what's playing
            return await self._handle_whats_playing()

        except Exception as e:
            logger.error(f"Music tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble controlling Spotify.",
                data={"error": str(e)}
            )

    def _is_pause_request(self, query: str) -> bool:
        return any(word in query for word in ["pause", "stop music", "stop playing", "stop the music"])

    def _is_skip_next(self, query: str) -> bool:
        return any(pattern in query for pattern in ["skip", "next song", "next track", "skip this"])

    def _is_skip_previous(self, query: str) -> bool:
        return any(pattern in query for pattern in ["previous", "go back", "last song", "last track"])

    def _is_whats_playing(self, query: str) -> bool:
        patterns = ["what's playing", "what is playing", "what song", "currently playing", "now playing"]
        return any(p in query for p in patterns)

    def _is_volume_request(self, query: str) -> bool:
        return "volume" in query

    def _is_play_request(self, query: str) -> bool:
        return "play" in query

    async def _handle_pause(self) -> VoiceToolResult:
        """Handle pause request."""
        spotify = self._get_spotify()
        success = await spotify.pause()

        if success:
            return VoiceToolResult(
                success=True,
                message="Music paused.",
                data={"action": "paused"}
            )
        return VoiceToolResult(
            success=False,
            message="Couldn't pause. Is Spotify playing on a device?",
            data={"action": "pause_failed"}
        )

    async def _handle_skip_next(self) -> VoiceToolResult:
        """Handle skip to next track."""
        spotify = self._get_spotify()
        success = await spotify.skip_next()

        if success:
            # Wait a moment and get new track info
            import asyncio
            await asyncio.sleep(0.5)
            state = await spotify.get_current_playback()

            if state and state.track:
                return VoiceToolResult(
                    success=True,
                    message=f"Playing {state.track.to_voice_summary()}",
                    data={"action": "skipped", "track": state.track.to_dict()}
                )
            return VoiceToolResult(
                success=True,
                message="Skipped to next track.",
                data={"action": "skipped"}
            )

        return VoiceToolResult(
            success=False,
            message="Couldn't skip. Is Spotify active?",
            data={"action": "skip_failed"}
        )

    async def _handle_skip_previous(self) -> VoiceToolResult:
        """Handle skip to previous track."""
        spotify = self._get_spotify()
        success = await spotify.skip_previous()

        if success:
            import asyncio
            await asyncio.sleep(0.5)
            state = await spotify.get_current_playback()

            if state and state.track:
                return VoiceToolResult(
                    success=True,
                    message=f"Playing {state.track.to_voice_summary()}",
                    data={"action": "previous", "track": state.track.to_dict()}
                )
            return VoiceToolResult(
                success=True,
                message="Playing previous track.",
                data={"action": "previous"}
            )

        return VoiceToolResult(
            success=False,
            message="Couldn't go back. Is Spotify active?",
            data={"action": "previous_failed"}
        )

    async def _handle_whats_playing(self) -> VoiceToolResult:
        """Handle what's playing request."""
        spotify = self._get_spotify()
        state = await spotify.get_current_playback()

        if not state:
            return VoiceToolResult(
                success=True,
                message="Nothing is currently playing on Spotify.",
                data={"playing": False}
            )

        if not state.is_playing:
            if state.track:
                return VoiceToolResult(
                    success=True,
                    message=f"Paused: {state.track.to_voice_summary()}",
                    data={"playing": False, "paused_track": state.track.to_dict()}
                )
            return VoiceToolResult(
                success=True,
                message="Spotify is paused.",
                data={"playing": False}
            )

        return VoiceToolResult(
            success=True,
            message=state.track.to_voice_summary(),
            data={"playing": True, "track": state.track.to_dict(), "device": state.device_name}
        )

    async def _handle_volume(self, query: str) -> VoiceToolResult:
        """Handle volume control."""
        spotify = self._get_spotify()

        # Extract volume level or direction
        if "up" in query:
            state = await spotify.get_current_playback()
            current_vol = state.volume_percent if state else 50
            new_vol = min(100, current_vol + 15)
            await spotify.set_volume(new_vol)
            return VoiceToolResult(
                success=True,
                message=f"Volume up to {new_vol}%.",
                data={"volume": new_vol}
            )

        if "down" in query:
            state = await spotify.get_current_playback()
            current_vol = state.volume_percent if state else 50
            new_vol = max(0, current_vol - 15)
            await spotify.set_volume(new_vol)
            return VoiceToolResult(
                success=True,
                message=f"Volume down to {new_vol}%.",
                data={"volume": new_vol}
            )

        # Look for specific percentage
        match = re.search(r'(\d+)\s*%?', query)
        if match:
            volume = int(match.group(1))
            await spotify.set_volume(volume)
            return VoiceToolResult(
                success=True,
                message=f"Volume set to {volume}%.",
                data={"volume": volume}
            )

        # Just report current volume
        state = await spotify.get_current_playback()
        if state:
            return VoiceToolResult(
                success=True,
                message=f"Volume is at {state.volume_percent}%.",
                data={"volume": state.volume_percent}
            )

        return VoiceToolResult(
            success=False,
            message="Couldn't get volume level.",
            data={}
        )

    async def _handle_play(self, query: str) -> VoiceToolResult:
        """Handle play request."""
        spotify = self._get_spotify()
        query_lower = query.lower()

        # Check for playlist request
        playlist_match = re.search(r'play\s+(?:my\s+)?(?:the\s+)?(.+?)\s+playlist', query_lower)
        if playlist_match or "playlist" in query_lower:
            playlist_name = playlist_match.group(1) if playlist_match else self._extract_play_query(query)
            if playlist_name:
                result = await spotify.play_playlist(playlist_name)
                if result:
                    return VoiceToolResult(
                        success=True,
                        message=f"Playing {result} playlist.",
                        data={"action": "play_playlist", "playlist": result}
                    )
                return VoiceToolResult(
                    success=False,
                    message=f"Couldn't find a playlist called {playlist_name}.",
                    data={"playlist_not_found": playlist_name}
                )

        # Check if just "play" (resume)
        play_query = self._extract_play_query(query)
        if not play_query or play_query in ["music", "something", "it"]:
            success = await spotify.play()
            if success:
                import asyncio
                await asyncio.sleep(0.5)
                state = await spotify.get_current_playback()
                if state and state.track:
                    return VoiceToolResult(
                        success=True,
                        message=f"Playing {state.track.to_voice_summary()}",
                        data={"action": "resumed", "track": state.track.to_dict()}
                    )
                return VoiceToolResult(
                    success=True,
                    message="Resuming playback.",
                    data={"action": "resumed"}
                )
            return VoiceToolResult(
                success=False,
                message="Couldn't start playback. Is Spotify active on a device?",
                data={"action": "play_failed"}
            )

        # Search and play
        track = await spotify.play_search(play_query)
        if track:
            return VoiceToolResult(
                success=True,
                message=f"Playing {track.to_voice_summary()}",
                data={"action": "play_search", "track": track.to_dict()}
            )

        return VoiceToolResult(
            success=False,
            message=f"Couldn't find '{play_query}' on Spotify.",
            data={"not_found": play_query}
        )

    def _extract_play_query(self, query: str) -> Optional[str]:
        """Extract what to play from query."""
        patterns = [
            r"play\s+(?:some\s+)?(.+?)(?:\s+on\s+spotify)?$",
            r"put\s+on\s+(.+)",
            r"listen\s+to\s+(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                # Remove trailing words like "please"
                result = re.sub(r'\s+(please|now|for me)$', '', result, flags=re.IGNORECASE)
                return result

        return None
