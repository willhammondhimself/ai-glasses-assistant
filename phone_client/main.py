"""
JARVIS Phone Client - Main Entry Point.
Runs on phone, controls Halo glasses via Bluetooth LE.
"""
import asyncio
import logging
import yaml
import time
from pathlib import Path
from typing import Optional

# Local imports
from halo.connection import HaloConnection, HaloConfig, scan_for_halo
from jarvis.personality import JarvisPersonality
from hud.renderer import HUDRenderer
from modes.mental_math import MentalMathMode, DrillConfig
from edith.scanner import EdithScanner

# Optional: Whisper for voice recognition
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JarvisClient:
    """
    Main JARVIS client.

    Orchestrates:
    - Bluetooth connection to Halo glasses
    - Voice recognition
    - Mode management (Mental Math, Poker, etc.)
    - EDITH background scanning
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.halo: Optional[HaloConnection] = None
        self.jarvis = JarvisPersonality(
            user_name=self.config["user"]["name"],
            address=self.config["user"]["address"],
            location=self.config["user"]["locations"][0] if self.config["user"]["locations"] else None,
            humor_level=self.config["personality"]["humor_level"]
        )
        self.renderer = HUDRenderer()
        self.edith: Optional[EdithScanner] = None

        # Current mode
        self.current_mode: Optional[MentalMathMode] = None
        self.mode_name: str = "idle"

        # Voice recognition
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            self._init_whisper()

        # State
        self.running = False
        self._input_queue: asyncio.Queue = asyncio.Queue()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(__file__).parent / config_path
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config not found at {path}, using defaults")
            return self._default_config()

    def _default_config(self) -> dict:
        """Default configuration."""
        return {
            "user": {
                "name": "Will",
                "address": "sir",
                "locations": ["Claremont, CA"],
                "target_speed": 0.25,
            },
            "hardware": {
                "halo_mac_address": "XX:XX:XX:XX:XX:XX",
            },
            "voice": {
                "model": "small.en",
                "wake_words": ["jarvis", "mental math"],
            },
            "personality": {
                "humor_level": "high",
            },
            "mental_math": {
                "time_targets": {1: 2000, 2: 4000, 3: 8000, 4: 12000, 5: 20000},
                "default_difficulty": 2,
            },
            "edith": {
                "enabled": True,
                "scan_interval_seconds": 5,
            },
        }

    def _init_whisper(self):
        """Initialize Whisper voice recognition."""
        try:
            model_name = self.config["voice"]["model"]
            logger.info(f"Loading Whisper model: {model_name}")
            self.whisper_model = whisper.load_model(model_name)
            logger.info("Whisper loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self.whisper_model = None

    async def connect(self) -> bool:
        """Connect to Halo glasses."""
        mac = self.config["hardware"]["halo_mac_address"]

        # If MAC is placeholder, scan for devices
        if mac == "XX:XX:XX:XX:XX:XX":
            logger.info("Scanning for Halo devices...")
            devices = await scan_for_halo(10.0)

            if not devices:
                logger.error("No Halo devices found")
                return False

            logger.info(f"Found: {devices}")
            mac = devices[0]["address"]

        # Create connection
        halo_config = HaloConfig(
            mac_address=mac,
            display_width=self.config["hardware"].get("display", {}).get("width", 640),
            display_height=self.config["hardware"].get("display", {}).get("height", 400),
        )

        self.halo = HaloConnection(halo_config)

        # Show connecting screen
        lua = self.renderer.render_connecting()
        logger.info("Connecting to Halo...")

        if await self.halo.connect():
            # Send connecting screen
            await self.halo.send_lua(lua)
            await asyncio.sleep(0.5)

            # Show greeting
            greeting = self.jarvis.get_greeting()
            await self._show_idle(greeting)
            await self.halo.speak(greeting, block=True)

            return True

        return False

    async def disconnect(self):
        """Disconnect from Halo."""
        if self.halo:
            await self.halo.disconnect()
            self.halo = None

    async def _send_display(self, lua: str) -> bool:
        """Send display update to Halo."""
        if self.halo and self.halo.is_connected:
            return await self.halo.send_lua(lua)
        return False

    async def _speak(self, text: str) -> bool:
        """Speak text via TTS."""
        if self.halo:
            return await self.halo.speak(text)
        return False

    async def _show_idle(self, message: str = "Ready"):
        """Show idle screen."""
        from datetime import datetime
        time_str = datetime.now().strftime("%H:%M")
        lua = self.renderer.render_idle(message, time_str)
        await self._send_display(lua)

    async def start_mental_math(self, difficulty: int = 2):
        """Start mental math mode."""
        config = DrillConfig(
            difficulty=difficulty,
            problem_count=None,  # Infinite
            show_result_ms=1500,
            auto_advance=True,
            speak_feedback=True,
        )

        self.current_mode = MentalMathMode(
            config=config,
            jarvis=self.jarvis,
            renderer=self.renderer,
            send_display=self._send_display,
            speak=self._speak,
        )

        self.mode_name = "mental_math"
        await self.current_mode.start()

    async def stop_mode(self):
        """Stop current mode."""
        if self.current_mode:
            if isinstance(self.current_mode, MentalMathMode):
                await self.current_mode.stop()
            self.current_mode = None
            self.mode_name = "idle"
            await self._show_idle("Mode ended")

    async def process_voice_command(self, text: str):
        """
        Process a voice command.

        Commands:
        - "mental math" / "start drill" - Start mental math
        - "stop" / "end" - Stop current mode
        - "difficulty X" - Change difficulty
        - "skip" - Skip current problem
        - [number] - Answer to current problem
        """
        text = text.lower().strip()
        logger.info(f"Voice command: '{text}'")

        # Check for wake words / mode activation
        if any(w in text for w in ["mental math", "start drill", "math mode"]):
            # Extract difficulty if specified
            difficulty = 2
            for d in range(1, 6):
                if f"difficulty {d}" in text or f"level {d}" in text:
                    difficulty = d
                    break

            await self.start_mental_math(difficulty)
            return

        # Stop commands
        if any(w in text for w in ["stop", "end", "quit", "exit"]):
            await self.stop_mode()
            return

        # Difficulty change
        if "difficulty" in text:
            for d in range(1, 6):
                if str(d) in text:
                    if isinstance(self.current_mode, MentalMathMode):
                        self.current_mode.set_difficulty(d)
                        await self._speak(f"Difficulty set to {d}")
                    return

        # Skip command
        if "skip" in text:
            if isinstance(self.current_mode, MentalMathMode):
                await self.current_mode.skip_problem()
            return

        # If in mental math mode, treat as answer
        if isinstance(self.current_mode, MentalMathMode) and self.current_mode.is_active:
            await self.current_mode.submit_answer(text)
            return

        # Unknown command
        logger.info(f"Unrecognized command: {text}")

    async def process_input(self, text: str):
        """Process any input (voice or keyboard)."""
        await self._input_queue.put(text)

    async def _input_processor(self):
        """Process inputs from queue."""
        while self.running:
            try:
                text = await asyncio.wait_for(self._input_queue.get(), timeout=0.1)
                await self.process_voice_command(text)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Input processing error: {e}")

    async def run(self):
        """Main run loop."""
        self.running = True

        # Start input processor
        input_task = asyncio.create_task(self._input_processor())

        # Start EDITH if enabled
        if self.config["edith"]["enabled"]:
            # EDITH would run in background
            pass

        logger.info("JARVIS client running. Type commands or speak.")

        try:
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            input_task.cancel()

    async def shutdown(self):
        """Shutdown the client."""
        self.running = False
        await self.stop_mode()
        await self.disconnect()
        logger.info("JARVIS client shutdown complete")


async def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                     J.A.R.V.I.S                           ║
    ║           Just A Rather Very Intelligent System           ║
    ║                                                           ║
    ║              Phone Client for Halo Glasses                ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    client = JarvisClient()

    # Try to connect to Halo
    print("Attempting to connect to Halo glasses...")
    connected = await client.connect()

    if not connected:
        print("\nCould not connect to Halo glasses.")
        print("Running in simulation mode (no glasses).\n")

        # Create mock connection for testing
        async def mock_send(lua):
            return True

        async def mock_speak(text):
            print(f"[JARVIS] {text}")
            return True

        client._send_display = mock_send
        client._speak = mock_speak

    print("\nCommands:")
    print("  'mental math'    - Start mental math drill")
    print("  'difficulty N'   - Set difficulty (1-5)")
    print("  'stop'           - Stop current mode")
    print("  'quit'           - Exit program")
    print("  [number]         - Answer current problem")
    print()

    # Run client in background
    client_task = asyncio.create_task(client.run())

    # Simple console input loop
    try:
        while client.running:
            # Non-blocking input
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, input, "> "
                )
                line = line.strip()

                if line.lower() in ["quit", "exit", "q"]:
                    break

                if line:
                    await client.process_input(line)

            except EOFError:
                break

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        await client.shutdown()
        client_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
