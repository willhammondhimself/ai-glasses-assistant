#!/usr/bin/env python3
"""
WHAM Meeting Mode Test Client

Interactive CLI client for testing Meeting Mode without Halo glasses hardware.

Usage:
    # Start the backend server first:
    uvicorn backend.server:app --reload

    # Then run this test client:
    python -m backend.meeting.test_client

Commands:
    start [type]    - Start meeting (type: general/negotiation/interview/sales)
    end             - End meeting session
    tap             - Simulate double-tap (quick suggestion)
    ask <question>  - Voice command with text query
    audio <file>    - Send audio file for transcription
    status          - Show session status
    transcript      - Show recent transcript
    help            - Show this help
    quit            - Exit client
"""

import asyncio
import base64
import json
import os
import sys
import uuid
import wave
import io
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


class MeetingTestClient:
    """Interactive test client for WHAM Meeting Mode."""

    DEFAULT_SERVER = "ws://localhost:8000"

    def __init__(self, server_url: str = None):
        self.server_url = server_url or self.DEFAULT_SERVER
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self.transcript: List[Dict[str, Any]] = []
        self.suggestions: List[Dict[str, Any]] = []
        self.running = False
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self, session_id: str = None):
        """Connect to the meeting WebSocket endpoint."""
        session_id = session_id or str(uuid.uuid4())[:8]
        endpoint = f"{self.server_url}/ws/meeting/{session_id}"

        try:
            self.ws = await websockets.connect(endpoint)
            self.session_id = session_id
            print(f"[CONNECTED] {endpoint}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            return False

    async def disconnect(self):
        """Disconnect from server."""
        if self.ws:
            await self.ws.close()
            self.ws = None
            print("[DISCONNECTED]")

    async def start_meeting(self, meeting_type: str = "general"):
        """Start a meeting session."""
        if not self.ws:
            print("[ERROR] Not connected. Use 'connect' first.")
            return

        await self.ws.send(json.dumps({
            "type": "meeting_start",
            "config": {
                "meeting_type": meeting_type,
                "participants": ["Will", "Counterpart"],
                "proactive_suggestions": False,
            },
        }))
        print(f"[SENT] meeting_start ({meeting_type})")

    async def end_meeting(self):
        """End the meeting session."""
        if not self.ws:
            print("[ERROR] Not connected.")
            return

        await self.ws.send(json.dumps({"type": "meeting_end"}))
        print("[SENT] meeting_end")

    async def double_tap(self):
        """Simulate double-tap gesture for quick suggestion."""
        if not self.ws:
            print("[ERROR] Not connected.")
            return

        await self.ws.send(json.dumps({"type": "double_tap"}))
        print("[SENT] double_tap (waiting for suggestion...)")

    async def voice_command(self, query: str):
        """Send voice command with text query."""
        if not self.ws:
            print("[ERROR] Not connected.")
            return

        await self.ws.send(json.dumps({
            "type": "voice_command",
            "query": query,
        }))
        print(f"[SENT] voice_command: {query}")

    async def send_audio_file(self, filepath: str):
        """Read WAV file and send as audio_chunk."""
        if not self.ws:
            print("[ERROR] Not connected.")
            return

        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            return

        try:
            with open(filepath, "rb") as f:
                audio_bytes = f.read()

            # Try to get WAV info
            try:
                with wave.open(filepath, "rb") as wav:
                    duration_ms = int(wav.getnframes() / wav.getframerate() * 1000)
                    sample_rate = wav.getframerate()
            except:
                duration_ms = 1000
                sample_rate = 16000

            audio_b64 = base64.b64encode(audio_bytes).decode()

            await self.ws.send(json.dumps({
                "type": "audio_chunk",
                "audio": audio_b64,
                "duration_ms": duration_ms,
                "sample_rate": sample_rate,
            }))
            print(f"[SENT] audio_chunk ({duration_ms}ms, {sample_rate}Hz)")

        except Exception as e:
            print(f"[ERROR] Failed to send audio: {e}")

    async def send_test_audio(self):
        """Generate and send a test audio chunk (silence)."""
        if not self.ws:
            print("[ERROR] Not connected.")
            return

        # Generate 1 second of silence
        sample_rate = 16000
        duration_ms = 1000
        num_samples = int(sample_rate * duration_ms / 1000)
        audio_data = bytes(num_samples * 2)

        # Wrap in WAV format
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_data)

        audio_b64 = base64.b64encode(buffer.getvalue()).decode()

        await self.ws.send(json.dumps({
            "type": "audio_chunk",
            "audio": audio_b64,
            "duration_ms": duration_ms,
            "sample_rate": sample_rate,
        }))
        print("[SENT] test audio chunk (1s silence)")

    def show_transcript(self, last_n: int = 10):
        """Show recent transcript entries."""
        if not self.transcript:
            print("[INFO] No transcript yet.")
            return

        print("\n--- Recent Transcript ---")
        for seg in self.transcript[-last_n:]:
            speaker = seg.get("speaker", "unknown")
            text = seg.get("text", "")
            print(f"  [{speaker}]: {text}")
        print("-------------------------\n")

    def show_status(self):
        """Show current session status."""
        print("\n--- Session Status ---")
        print(f"  Server: {self.server_url}")
        print(f"  Session ID: {self.session_id or 'None'}")
        print(f"  Connected: {self.ws is not None}")
        print(f"  Transcript segments: {len(self.transcript)}")
        print(f"  Suggestions received: {len(self.suggestions)}")
        print("----------------------\n")

    async def _receive_messages(self):
        """Background task to receive and display messages."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type", "unknown")

                if msg_type == "status":
                    status = data.get("status", "")
                    message_text = data.get("message", "")
                    session_id = data.get("session_id", "")

                    if session_id:
                        self.session_id = session_id

                    if status == "meeting_started":
                        print(f"\n[STATUS] Meeting started | Session: {session_id}")
                        print(f"         {message_text}")
                    elif status == "meeting_ended":
                        summary = data.get("summary", {})
                        duration = summary.get("duration_seconds", 0)
                        suggestions = summary.get("suggestions_given", 0)
                        print(f"\n[STATUS] Meeting ended")
                        print(f"         Duration: {duration:.0f}s | Suggestions: {suggestions}")
                    elif status == "processing":
                        print(f"\n[PROCESSING] {message_text}")
                    else:
                        print(f"\n[STATUS] {status}: {message_text}")

                elif msg_type == "transcript_update":
                    segment = data.get("segment", {})
                    self.transcript.append(segment)
                    speaker = segment.get("speaker", "?")
                    text = segment.get("text", "")
                    print(f"\n[TRANSCRIPT] [{speaker}]: {text}")

                elif msg_type == "suggestion":
                    self.suggestions.append(data)
                    trigger = data.get("trigger", "?")
                    suggestion_data = data.get("suggestion", {})
                    suggestion_text = suggestion_data.get("suggestion", "")
                    latency = data.get("total_latency_ms", 0)

                    print(f"\n[SUGGESTION] ({trigger}, {latency:.0f}ms)")
                    print(f"  {suggestion_text}")

                    alternatives = suggestion_data.get("alternatives", [])
                    if alternatives:
                        print("  Alternatives:")
                        for alt in alternatives[:2]:
                            print(f"    - {alt}")

                    tactical = suggestion_data.get("tactical_notes")
                    if tactical:
                        print(f"  TIP: {tactical}")

                elif msg_type == "error":
                    error_msg = data.get("message", "Unknown error")
                    print(f"\n[ERROR] {error_msg}")

                elif msg_type == "pong":
                    print("[PONG] Heartbeat OK")

                else:
                    print(f"\n[UNKNOWN] {data}")

                # Show prompt again
                print("\nwham> ", end="", flush=True)

        except websockets.exceptions.ConnectionClosed:
            print("\n[DISCONNECTED] Connection closed by server")
        except Exception as e:
            print(f"\n[ERROR] Receive error: {e}")

    def print_help(self):
        """Print help message."""
        print("""
WHAM Meeting Mode Test Client
=============================

Commands:
  start [type]    Start meeting (general/negotiation/interview/sales)
  end             End the meeting session
  tap             Simulate double-tap for quick suggestion
  ask <question>  Send voice command with text query
  audio <file>    Send audio file (.wav) for transcription
  testaudio       Send 1s test audio (silence)
  status          Show session status
  transcript      Show recent transcript
  help            Show this help
  quit            Exit client

Examples:
  wham> start negotiation
  wham> tap
  wham> ask What leverage do I have here?
  wham> end
""")

    async def run_interactive(self):
        """Main interactive REPL loop."""
        print("\n" + "=" * 50)
        print("  WHAM Meeting Mode Test Client")
        print("  Type 'help' for commands, 'quit' to exit")
        print("=" * 50 + "\n")

        # Connect to server
        if not await self.connect():
            return

        # Start background receiver
        self._receive_task = asyncio.create_task(self._receive_messages())
        self.running = True

        try:
            while self.running:
                try:
                    # Get user input
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("wham> ")
                    )
                except EOFError:
                    break

                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "help":
                    self.print_help()
                elif cmd == "start":
                    meeting_type = args or "general"
                    await self.start_meeting(meeting_type)
                elif cmd == "end":
                    await self.end_meeting()
                elif cmd == "tap":
                    await self.double_tap()
                elif cmd == "ask":
                    if not args:
                        print("[ERROR] Usage: ask <question>")
                    else:
                        await self.voice_command(args)
                elif cmd == "audio":
                    if not args:
                        print("[ERROR] Usage: audio <filepath>")
                    else:
                        await self.send_audio_file(args)
                elif cmd == "testaudio":
                    await self.send_test_audio()
                elif cmd == "status":
                    self.show_status()
                elif cmd == "transcript":
                    self.show_transcript()
                elif cmd == "ping":
                    if self.ws:
                        await self.ws.send(json.dumps({"type": "ping"}))
                        print("[SENT] ping")
                else:
                    print(f"[ERROR] Unknown command: {cmd}")
                    print("  Type 'help' for available commands")

        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")

        finally:
            self.running = False
            if self._receive_task:
                self._receive_task.cancel()
            await self.disconnect()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="WHAM Meeting Mode Test Client")
    parser.add_argument(
        "--server",
        default="ws://localhost:8000",
        help="WebSocket server URL (default: ws://localhost:8000)",
    )
    args = parser.parse_args()

    client = MeetingTestClient(server_url=args.server)
    await client.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
