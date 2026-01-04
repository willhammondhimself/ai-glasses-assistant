"""
WHAM Phone Client - Main Entry Point.
Will Hammond's Augmented Mind.
Runs on phone, controls Halo glasses via Bluetooth LE.
"""
import asyncio
import logging
import yaml
import time
import os
from pathlib import Path
from typing import Optional, Union

# Local imports
from halo.connection import HaloConnection, HaloConfig, scan_for_halo
from wham.personality import WHAMPersonality
from hud.renderer import HUDRenderer
from hud.poker_display import PokerHUD
from hud.homework_display import HomeworkHUD
from hud.code_debug_display import CodeDebugHUD
from modes.mental_math import MentalMathMode, DrillConfig
from modes.poker_coach import LivePokerCoach
from modes.homework_mode import HomeworkMode, HomeworkConfig
from modes.code_debug_mode import CodeDebugMode, DebugConfig
from edith.scanner import EdithScanner, ScanConfig
from core.router import IntelligenceRouter

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


class WHAMClient:
    """
    Main WHAM client.

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
        self.wham = WHAMPersonality(
            user_name=self.config["user"]["name"],
            address=self.config["user"].get("address", ""),
            location=self.config["user"]["locations"][0] if self.config["user"]["locations"] else None,
            humor_level=self.config["personality"].get("humor_level", "low")
        )
        self.renderer = HUDRenderer()
        self.poker_hud = PokerHUD()
        self.homework_hud = HomeworkHUD()
        self.code_debug_hud = CodeDebugHUD()
        self.edith: Optional[EdithScanner] = None
        self.router = IntelligenceRouter()

        # Current mode
        self.current_mode: Optional[Union[MentalMathMode, LivePokerCoach, HomeworkMode, CodeDebugMode]] = None
        self.mode_name: str = "idle"

        # Mode instances
        self.poker_coach: Optional[LivePokerCoach] = None
        self.homework_mode: Optional[HomeworkMode] = None
        self.code_debug_mode: Optional[CodeDebugMode] = None

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
                "address": "",
                "locations": ["Claremont, CA"],
                "target_speed": 0.25,
            },
            "hardware": {
                "halo_mac_address": "XX:XX:XX:XX:XX:XX",
            },
            "voice": {
                "model": "small.en",
                "wake_words": ["wham", "hey wham", "mental math", "poker"],
            },
            "personality": {
                "name": "WHAM",
                "full_name": "Will Hammond's Augmented Mind",
                "humor_level": "low",
            },
            "mental_math": {
                "time_targets": {1: 2000, 2: 4000, 3: 8000, 4: 12000, 5: 20000},
                "default_difficulty": 2,
            },
            "poker": {
                "stakes": "$0.25/$0.50",
                "bb_value": 0.50,
                "session_dir": "./poker_sessions",
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
            greeting = self.wham.get_greeting()
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
            wham=self.wham,
            renderer=self.renderer,
            send_display=self._send_display,
            speak=self._speak,
        )

        self.mode_name = "mental_math"
        await self.current_mode.start()

    async def start_poker_coach(self):
        """Start live poker coaching mode."""
        poker_config = self.config.get("poker", {})

        self.poker_coach = LivePokerCoach(
            stakes=poker_config.get("stakes", "$0.25/$0.50"),
            bb_value=poker_config.get("bb_value", 0.50),
            persist_dir=poker_config.get("session_dir", "./poker_sessions")
        )

        await self.poker_coach.start()
        self.current_mode = self.poker_coach
        self.mode_name = "poker"

        await self._speak("Poker coach online. Show cards.")
        logger.info("Poker coach started")

    async def start_homework_mode(self):
        """Start homework assistance mode."""
        config = HomeworkConfig(
            speak_solution=True,
            show_steps=True,
            auto_advance=True,
            cloud_fallback=True,
        )

        self.homework_mode = HomeworkMode(
            config=config,
            wham=self.wham,
            renderer=self.renderer,
            send_display=self._send_display,
            speak=self._speak,
        )

        await self.homework_mode.start()
        self.current_mode = self.homework_mode
        self.mode_name = "homework"

        logger.info("Homework mode started")

    async def start_code_debug_mode(self):
        """Start code debugging mode."""
        config = DebugConfig(
            speak_errors=True,
            auto_explain=False,
            show_fix=True,
            cloud_explain=True,
        )

        self.code_debug_mode = CodeDebugMode(
            config=config,
            wham=self.wham,
            renderer=self.renderer,
            send_display=self._send_display,
            speak=self._speak,
        )

        await self.code_debug_mode.start()
        self.current_mode = self.code_debug_mode
        self.mode_name = "debug"

        logger.info("Code debug mode started")

    async def analyze_homework(self, text: str = ""):
        """Analyze math problem - via image or text."""
        if not self.homework_mode:
            await self.start_homework_mode()

        # Try image capture first
        if self.halo:
            image = await self.halo.capture_image()
            if image:
                # Show thinking
                lua = self.homework_hud.render_thinking(
                    problem="Extracting...",
                    elapsed_s=0,
                    method="gemini_ocr"
                )
                await self._send_display(lua)

                # Analyze
                result = await self.homework_mode.analyze(image)

                # Show solution
                lua = self.homework_hud.render_solution(
                    problem=result.problem,
                    answer=result.answer,
                    steps=result.steps,
                    method=result.method,
                    cost=result.cost,
                    latency_ms=result.latency_ms
                )
                await self._send_display(lua)
                return

        # Text-based fallback
        if text:
            result = await self.homework_mode.solve_text(text)

            lua = self.homework_hud.render_solution(
                problem=result.problem,
                answer=result.answer,
                steps=result.steps,
                method=result.method,
                cost=result.cost,
                latency_ms=result.latency_ms
            )
            await self._send_display(lua)

    async def analyze_code(self, text: str = ""):
        """Analyze code for errors - via image or text."""
        if not self.code_debug_mode:
            await self.start_code_debug_mode()

        # Try image capture first
        if self.halo:
            image = await self.halo.capture_image()
            if image:
                # Show thinking
                lua = self.code_debug_hud.render_analyzing(
                    elapsed_s=0,
                    language="detecting"
                )
                await self._send_display(lua)

                # Analyze
                result = await self.code_debug_mode.analyze(image)

                # Show result
                if result.valid:
                    lua = self.code_debug_hud.render_syntax_ok(
                        language=result.language,
                        latency_ms=result.latency_ms
                    )
                else:
                    errors = [
                        {"line": e.line, "message": e.message, "suggestion": e.suggestion}
                        for e in result.errors
                    ]
                    lua = self.code_debug_hud.render_errors(
                        language=result.language,
                        errors=errors,
                        latency_ms=result.latency_ms,
                        cost=result.cost
                    )
                await self._send_display(lua)
                return

        # Text-based fallback
        if text:
            result = await self.code_debug_mode.check_text(text)

            if result.valid:
                lua = self.code_debug_hud.render_syntax_ok(
                    language=result.language,
                    latency_ms=result.latency_ms
                )
            else:
                errors = [
                    {"line": e.line, "message": e.message, "suggestion": e.suggestion}
                    for e in result.errors
                ]
                lua = self.code_debug_hud.render_errors(
                    language=result.language,
                    errors=errors,
                    latency_ms=result.latency_ms,
                    cost=result.cost
                )
            await self._send_display(lua)

    async def explain_error(self):
        """Get detailed explanation for current error."""
        if not self.code_debug_mode:
            await self._speak("Start debug mode first.")
            return

        explanation = await self.code_debug_mode.explain_error(0)
        await self._speak(explanation[:200])

        # Show on HUD
        if self.code_debug_mode.current_result and self.code_debug_mode.current_result.errors:
            error = self.code_debug_mode.current_result.errors[0]
            lua = self.code_debug_hud.render_explanation(
                error={"line": error.line, "message": error.message},
                explanation=explanation,
                cost=0.007
            )
            await self._send_display(lua)

    async def get_hint(self):
        """Get hint for current homework problem."""
        if not self.homework_mode:
            await self._speak("Start homework mode first.")
            return

        hint = await self.homework_mode.get_hint()
        await self._speak(hint)

        if self.homework_mode.current_problem:
            lua = self.homework_hud.render_hint(
                problem=self.homework_mode.current_problem,
                hint=hint,
                cost=0.007
            )
            await self._send_display(lua)

    async def stop_mode(self):
        """Stop current mode."""
        if self.current_mode:
            if isinstance(self.current_mode, MentalMathMode):
                await self.current_mode.stop()
            elif isinstance(self.current_mode, LivePokerCoach):
                summary = await self.current_mode.stop()
                profit = summary.get("profit_bb", 0)
                await self._speak(f"Session ended. {profit:+.1f} big blinds.")
            elif isinstance(self.current_mode, HomeworkMode):
                summary = await self.current_mode.stop()
                problems = summary.get("problems_solved", 0)
                cost = summary.get("total_cost", 0)
                await self._speak(f"Homework session ended. {problems} problems solved. Cost: {cost:.3f} dollars.")
            elif isinstance(self.current_mode, CodeDebugMode):
                summary = await self.current_mode.stop()
                checked = summary.get("code_checked", 0)
                cost = summary.get("total_cost", 0)
                await self._speak(f"Debug session ended. {checked} code blocks checked. Cost: {cost:.3f} dollars.")

            self.current_mode = None
            self.mode_name = "idle"
            await self._show_idle("Mode ended")

    async def analyze_poker_hand(self, text: str = ""):
        """Analyze current poker hand."""
        if not self.poker_coach:
            await self._speak("Start poker mode first.")
            return

        # If we have camera, use image analysis
        if self.halo:
            image = await self.halo.capture_image()
            if image:
                # Show thinking
                lua = self.poker_hud.render_thinking(
                    hero_cards=["?", "?"],
                    elapsed_s=0
                )
                await self._send_display(lua)

                # Analyze
                rec = await self.poker_coach.analyze_hand(image)

                # Show recommendation
                lua = self.poker_hud.render_recommendation(
                    hero_cards=rec.villain_type,  # Placeholder
                    board=[],
                    action=rec.action,
                    sizing=rec.sizing,
                    equity=rec.equity,
                    reasoning=rec.reasoning,
                    cost=rec.cost,
                    latency_ms=rec.latency_ms
                )
                await self._send_display(lua)
                await self._speak(f"{rec.action}. {rec.reasoning}")
                return

        # Text-based analysis fallback
        if text:
            # Parse "Ah Kc on 9s 7h 3d"
            parts = text.replace(" on ", "|").split("|")
            hero = parts[0] if parts else "Ah Kc"
            board = parts[1] if len(parts) > 1 else ""

            rec = await self.poker_coach.analyze_text(
                hero_cards=hero,
                board=board
            )

            await self._speak(f"{rec.action}. {rec.reasoning}")

    async def process_voice_command(self, text: str):
        """
        Process a voice command.

        Commands:
        - "mental math" / "math mode" - Start mental math
        - "poker" / "poker coach" - Start poker mode
        - "homework" / "solve" - Start homework mode / solve math
        - "debug" / "check code" - Start code debug mode / check code
        - "analyze" / "what should I do" - Analyze poker hand
        - "explain error" - Explain current code error
        - "hint" / "give hint" - Get hint for homework
        - "stop" / "end" - Stop current mode
        - "difficulty X" - Change difficulty
        - "skip" - Skip current problem
        - "villain [type]" - Set villain type
        - [number] - Answer to current problem
        """
        text = text.lower().strip()
        logger.info(f"Command: '{text}'")

        # Mental math activation
        if any(w in text for w in ["mental math", "math mode", "speed math"]):
            difficulty = 2
            for d in range(1, 6):
                if f"difficulty {d}" in text or f"level {d}" in text:
                    difficulty = d
                    break
            await self.start_mental_math(difficulty)
            return

        # Poker activation
        if any(w in text for w in ["poker", "poker coach", "live poker"]):
            await self.start_poker_coach()
            return

        # Homework mode activation
        if any(w in text for w in ["homework", "homework mode", "math help"]):
            await self.start_homework_mode()
            return

        # Solve math problem
        if any(w in text for w in ["solve", "calculate", "what is", "what's"]):
            # Extract the problem from the command
            problem = text
            for prefix in ["solve", "calculate", "what is", "what's"]:
                if problem.startswith(prefix):
                    problem = problem[len(prefix):].strip()
            await self.analyze_homework(problem)
            return

        # Code debug mode activation
        if any(w in text for w in ["debug mode", "code debug", "debug code"]):
            await self.start_code_debug_mode()
            return

        # Check code
        if any(w in text for w in ["check code", "syntax check", "check syntax"]):
            await self.analyze_code(text)
            return

        # Explain error
        if any(w in text for w in ["explain error", "why error", "what's wrong"]):
            await self.explain_error()
            return

        # Get hint
        if any(w in text for w in ["hint", "give hint", "help me"]):
            await self.get_hint()
            return

        # Poker analysis
        if any(w in text for w in ["analyze", "what should", "recommend", "action"]):
            if self.mode_name == "poker" or self.poker_coach:
                await self.analyze_poker_hand(text)
            return

        # Villain tracking
        if "villain" in text:
            if self.poker_coach:
                if "calling station" in text:
                    self.poker_coach.opponent_tracker.set_villain("v1", "Calling Station")
                    await self._speak("Tracking calling station.")
                elif "tight" in text:
                    self.poker_coach.opponent_tracker.set_villain("v1", "Tight Player")
                    await self._speak("Tracking tight player.")
                elif "maniac" in text:
                    self.poker_coach.opponent_tracker.set_villain("v1", "Maniac")
                    await self._speak("Tracking maniac.")
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
                        await self._speak(f"Difficulty {d}.")
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
        logger.info(f"Unrecognized: {text}")

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
                logger.error(f"Input error: {e}")

    async def run(self):
        """Main run loop."""
        self.running = True

        # Start input processor
        input_task = asyncio.create_task(self._input_processor())

        # Start EDITH if enabled
        if self.config["edith"]["enabled"]:
            pass  # EDITH runs in background

        logger.info("WHAM client running.")

        try:
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
        logger.info("WHAM shutdown complete")


# Backwards compatibility alias
JarvisClient = WHAMClient


async def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                        W.H.A.M                            ║
    ║            Will Hammond's Augmented Mind                  ║
    ║                                                           ║
    ║              Phone Client for Halo Glasses                ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    client = WHAMClient()

    # Try to connect to Halo
    print("Connecting to Halo glasses...")
    connected = await client.connect()

    if not connected:
        print("\nNo Halo connection. Simulation mode.\n")

        async def mock_send(lua):
            return True

        async def mock_speak(text):
            print(f"[WHAM] {text}")
            return True

        client._send_display = mock_send
        client._speak = mock_speak

    print("\nCommands:")
    print("  'mental math'    - Speed math drill")
    print("  'poker'          - Live poker coach")
    print("  'homework'       - Math homework helper")
    print("  'debug mode'     - Code syntax checker")
    print("  'solve X'        - Solve math problem X")
    print("  'check code'     - Check code for errors")
    print("  'explain error'  - Explain current error")
    print("  'hint'           - Get homework hint")
    print("  'analyze'        - Analyze poker hand")
    print("  'difficulty N'   - Set difficulty (1-5)")
    print("  'stop'           - Stop current mode")
    print("  'quit'           - Exit")
    print("  [number]         - Answer math problem")
    print()

    # Run client
    client_task = asyncio.create_task(client.run())

    try:
        while client.running:
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
