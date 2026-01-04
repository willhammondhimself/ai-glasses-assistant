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
from halo.oled_renderer import OLEDRenderer, create_oled_renderer
from halo.animations import AnimationEngine, create_animation_engine
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
from core.session_summary import SessionSummaryManager
from core.power_manager import PowerManager, PowerMode
from core.notifications import NotificationManager, NotificationPriority, Notification
from core.context_memory import ContextMemory
from core.daily_challenge import DailyChallenges, ChallengeCategory
from modes.morning_briefing import MorningAssistant, BriefingItemType, BriefingItem
from modes.quick_capture import QuickCapture, CaptureType
from modes.focus_mode import FocusMode, DistractionLevel

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
        # Power management (initialized before renderers)
        self.power_manager = PowerManager(self.config)
        self.power_manager.on_mode_change(self._on_power_mode_change)

        # Renderers (old and new)
        self.renderer = HUDRenderer()
        self.oled_renderer = create_oled_renderer(self.config, self.power_manager)
        self.animations = create_animation_engine(self.config, self.oled_renderer)

        # HUDs (now with power manager)
        self.poker_hud = PokerHUD(power_manager=self.power_manager, config=self.config)
        self.homework_hud = HomeworkHUD()
        self.code_debug_hud = CodeDebugHUD()

        # Notification system
        self.notifications = NotificationManager(self.config)
        self.notifications.set_display_callback(self._display_notification)

        # EDITH and router
        self.edith: Optional[EdithScanner] = None
        self.router = IntelligenceRouter()

        # Do Not Disturb state
        self._do_not_disturb = False

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

        # Session summary tracking
        self.session_summary = SessionSummaryManager(self.config)
        self.last_poker_result: Optional[dict] = None
        self.last_homework_result: Optional[dict] = None

        # Daily-life features
        self.context_memory = ContextMemory(self.config)
        self.daily_challenges = DailyChallenges(self.config)
        self.morning_assistant = MorningAssistant(
            self.config, user_name=self.config["user"]["name"]
        )
        self.quick_capture = QuickCapture(self.config)
        self.focus_mode = FocusMode(self.config)

        # Register challenge callbacks
        self.daily_challenges.on_challenge_complete(self._on_challenge_complete)
        self.daily_challenges.on_level_up(self._on_level_up)

        # Voice command shortcuts (batch commands)
        self._voice_shortcuts = {
            "work mode": ["focus on work", "do not disturb", "hide notifications"],
            "study mode": ["focus on studying", "do not disturb"],
            "meeting mode": ["start meeting mode", "do not disturb"],
            "break time": ["end focus", "notifications on"],
            "good morning": ["morning briefing"],
            "end day": ["show stats", "save session", "end focus"],
        }

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

    # ============================================================
    # Power & Notification Callbacks
    # ============================================================

    def _on_power_mode_change(self, old_mode: PowerMode, new_mode: PowerMode):
        """Handle power mode transitions."""
        logger.info(f"Power mode changed: {old_mode.value} → {new_mode.value}")

        # Notify user of significant changes
        if new_mode == PowerMode.CRITICAL:
            self.notifications.notify_battery_low(
                self.power_manager._state.battery_percent
            )
        elif new_mode == PowerMode.SAVER:
            self.notifications.push_quick(
                "Power Saver Active",
                "Display dimmed to save battery",
                NotificationPriority.MEDIUM,
                category="battery"
            )

        # Adjust EDITH behavior based on power
        if self.edith:
            if new_mode == PowerMode.CRITICAL:
                self.edith.set_battery_saver(True)
            elif old_mode == PowerMode.CRITICAL:
                self.edith.set_battery_saver(False)

    async def _display_notification(self, notification: Notification):
        """Display a notification on the HUD."""
        if not self.halo or not self.halo.connected:
            logger.debug(f"Cannot display notification (not connected): {notification.title}")
            return

        # Generate notification Lua code
        self.oled_renderer.clear()

        # Toast-style notification at bottom
        toast_height = 80
        toast_y = self.oled_renderer.height - toast_height

        # Background
        self.oled_renderer.rect(0, toast_y, self.oled_renderer.width, toast_height,
                                 (30, 30, 35), filled=True)

        # Color accent bar
        self.oled_renderer.rect(0, toast_y, 4, toast_height, notification.color, filled=True)

        # Icon if present
        text_x = 30
        if notification.icon:
            self.oled_renderer.text(notification.icon, 20, toast_y + 20, notification.color, 24)
            text_x = 50

        # Title
        self.oled_renderer.text(notification.title, text_x, toast_y + 20, notification.color, 24, "left")

        # Message
        if notification.message:
            msg = notification.message[:50] + "..." if len(notification.message) > 50 else notification.message
            self.oled_renderer.text(msg, text_x, toast_y + 50, (180, 180, 180), 18, "left")

        self.oled_renderer.show()

        try:
            await self.halo.send_lua(self.oled_renderer.get_lua())

            # Auto-dismiss after duration
            await asyncio.sleep(notification.duration_ms / 1000)
            self.notifications.dismiss_current()

        except Exception as e:
            logger.error(f"Failed to display notification: {e}")

    def _on_challenge_complete(self, challenge, xp_earned: int):
        """Handle challenge completion."""
        logger.info(f"Challenge completed: {challenge.title} (+{xp_earned} XP)")
        self.notifications.push_quick(
            f"Challenge Complete! +{xp_earned} XP",
            challenge.title,
            NotificationPriority.HIGH,
            category="challenge"
        )

    def _on_level_up(self, old_level: int, new_level: int):
        """Handle level up event."""
        logger.info(f"Level up: {old_level} → {new_level}")
        self.notifications.push_quick(
            f"Level Up! Level {new_level}",
            "Keep up the great work!",
            NotificationPriority.HIGH,
            category="achievement"
        )

    def update_battery(self, percent: int, charging: bool = False):
        """Update battery status from glasses."""
        mode_changed = self.power_manager.update_battery(percent, charging)

        # Update EDITH interval if needed
        if self.edith and mode_changed:
            if self.power_manager.is_low_power():
                self.edith.set_battery_saver(True)
            else:
                self.edith.set_battery_saver(False)

    def set_do_not_disturb(self, enabled: bool):
        """Toggle do not disturb mode."""
        self._do_not_disturb = enabled
        self.notifications.set_do_not_disturb(enabled)

        if enabled:
            logger.info("Do Not Disturb: ENABLED")
        else:
            logger.info("Do Not Disturb: DISABLED")

    def set_brightness(self, level: float):
        """Set display brightness (0.0-1.0)."""
        level = max(0.0, min(1.0, level))
        self.oled_renderer.set_brightness(level)
        logger.info(f"Brightness set to {level:.0%}")

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

    async def show_session_stats(self):
        """Display current session statistics on HUD."""
        summary = self.session_summary.generate_summary()
        formatted = self.session_summary.format_summary(summary)

        # Display on HUD (10 second duration)
        lua = self.renderer.render_text(formatted, title="SESSION STATS")
        await self._send_display(lua)

        # Speak brief summary
        await self._speak(
            f"Session: ${summary.total_cost:.2f} spent. "
            f"{summary.poker.count} poker hands. "
            f"{summary.homework.count} homework problems."
        )

    async def recall_last_result(self):
        """Show the last decision/result."""
        if self.last_poker_result:
            # Re-display last poker recommendation
            lua = self.poker_hud.render_recommendation(
                hero_cards=self.last_poker_result.get('hero_cards', ['?', '?']),
                board=self.last_poker_result.get('board', []),
                action=self.last_poker_result.get('action', 'FOLD'),
                sizing=self.last_poker_result.get('sizing', ''),
                equity=self.last_poker_result.get('equity', 0),
                reasoning=self.last_poker_result.get('reasoning', ''),
                cost=self.last_poker_result.get('cost', 0),
                confidence=self.last_poker_result.get('confidence')
            )
            await self._send_display(lua)
            await self._speak(f"Last hand: {self.last_poker_result.get('action', 'FOLD')}")
        elif self.last_homework_result:
            # Re-display last homework solution
            lua = self.homework_hud.render_solution(
                problem=self.last_homework_result.get('problem', ''),
                answer=self.last_homework_result.get('answer', ''),
                steps=self.last_homework_result.get('steps', []),
                method=self.last_homework_result.get('method', 'unknown'),
                cost=self.last_homework_result.get('cost', 0),
                latency_ms=self.last_homework_result.get('latency_ms', 0)
            )
            await self._send_display(lua)
            await self._speak(f"Last answer: {self.last_homework_result.get('answer', 'unknown')}")
        else:
            await self._speak("No recent results to recall.")

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
                    latency_ms=rec.latency_ms,
                    confidence=rec.confidence
                )
                await self._send_display(lua)
                await self._speak(f"{rec.action}. {rec.reasoning}")

                # Store for recall
                self.last_poker_result = {
                    'hero_cards': rec.villain_type,  # Placeholder
                    'board': [],
                    'action': rec.action,
                    'sizing': rec.sizing,
                    'equity': rec.equity,
                    'reasoning': rec.reasoning,
                    'cost': rec.cost,
                    'confidence': rec.confidence
                }

                # Record to session summary
                self.session_summary.record_poker_hand({
                    'hand_num': self.poker_coach.session.stats.hands_played if self.poker_coach.session else 0,
                    'cost': rec.cost,
                    'mistake_detected': False,
                    'mistake': None
                })
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

        # === BATCH VOICE SHORTCUTS ===
        for shortcut, commands in self._voice_shortcuts.items():
            if shortcut in text:
                logger.info(f"Executing batch shortcut: {shortcut}")
                for cmd in commands:
                    await self.process_voice_command(cmd)
                return

        # === MORNING BRIEFING ===
        if any(w in text for w in ["morning briefing", "good morning", "what's today"]):
            async def show_briefing(briefing):
                lines = self.morning_assistant.format_for_display(briefing)
                lua = self.renderer.render_text("\n".join(lines), title="MORNING BRIEFING")
                await self._send_display(lua)

            briefing = await self.morning_assistant.deliver_briefing(show_briefing)
            tts = self.morning_assistant.format_for_tts(briefing)
            await self._speak(tts)
            return

        # === QUICK CAPTURE ===
        if any(w in text for w in ["capture", "note", "remember that", "remind me"]):
            # Extract content after trigger
            content = text
            for trigger in ["capture", "note", "remember that", "remind me"]:
                if trigger in text:
                    idx = text.find(trigger)
                    content = text[idx + len(trigger):].strip()
                    break

            if content:
                capture = self.quick_capture.capture_voice(content)
                await self._speak(f"Captured: {capture.type.value}")

                # Record to challenges
                self.daily_challenges.record_progress(ChallengeCategory.PRODUCTIVITY, 1)
                return

        if "show captures" in text or "my notes" in text:
            recent = self.quick_capture.get_recent(5)
            if recent:
                lines = ["Recent Captures:"]
                for c in recent:
                    lines.append(f"• {c.content[:40]}...")
                lua = self.renderer.render_text("\n".join(lines), title="CAPTURES")
                await self._send_display(lua)
                await self._speak(f"You have {len(recent)} recent captures.")
            else:
                await self._speak("No captures yet.")
            return

        # === FOCUS MODE ===
        if any(w in text for w in ["focus on", "start focus", "pomodoro"]):
            # Extract task
            task = "work"
            for trigger in ["focus on", "start focus on"]:
                if trigger in text:
                    task = text.split(trigger)[-1].strip() or "work"
                    break

            session = await self.focus_mode.start_session(
                task=task,
                distraction_level=DistractionLevel.MEDIUM
            )
            await self._speak(f"Focus session started. {session.focus_duration_min} minutes on {task}.")

            # Show on HUD
            lines = self.focus_mode.format_for_display()
            lua = self.renderer.render_text("\n".join(lines), title="FOCUS MODE")
            await self._send_display(lua)
            return

        if any(w in text for w in ["end focus", "stop focus", "done focusing"]):
            if self.focus_mode.get_current_session():
                session = await self.focus_mode.end_session()
                if session:
                    # Record to challenges
                    self.daily_challenges.record_progress(
                        ChallengeCategory.FOCUS,
                        session.completed_pomodoros
                    )
                    await self._speak(
                        f"Focus ended. {session.completed_pomodoros} pomodoros, "
                        f"{session.total_focus_time_s // 60} minutes focused."
                    )
            else:
                await self._speak("No focus session active.")
            return

        if "skip break" in text:
            await self.focus_mode.skip_break()
            await self._speak("Break skipped. Focusing.")
            return

        if "focus status" in text:
            tts = self.focus_mode.format_for_tts()
            await self._speak(tts)
            return

        # === CONTEXT MEMORY ===
        if any(w in text for w in ["remember that", "keep in mind"]):
            # Parse and store memory
            memory = self.context_memory.parse_remember_command(text)
            if memory:
                await self._speak(f"I'll remember that about {memory.entity or 'this'}.")
                self.daily_challenges.record_progress(ChallengeCategory.LEARNING, 1)
            return

        if any(w in text for w in ["what do you know about", "tell me about", "recall"]):
            # Parse and recall
            memories = self.context_memory.parse_recall_command(text)
            if memories:
                lines = self.context_memory.format_memories_for_display(memories)
                lua = self.renderer.render_text("\n".join(lines), title="MEMORIES")
                await self._send_display(lua)
                await self._speak(f"Found {len(memories)} related memories.")
            else:
                await self._speak("I don't have any memories about that.")
            return

        # === DAILY CHALLENGES ===
        if any(w in text for w in ["challenges", "daily challenges", "my challenges"]):
            challenges = self.daily_challenges.get_daily_challenges()
            lines = self.daily_challenges.format_for_display()
            lua = self.renderer.render_text("\n".join(lines), title="DAILY CHALLENGES")
            await self._send_display(lua)
            tts = self.daily_challenges.format_for_tts()
            await self._speak(tts)
            return

        if any(w in text for w in ["my level", "xp", "progress"]):
            progress = self.daily_challenges.get_progress()
            await self._speak(
                f"Level {progress.level} with {progress.total_xp} XP. "
                f"{self.daily_challenges.get_xp_to_next_level()} XP to next level. "
                f"{progress.current_streak} day streak."
            )
            return

        # === VOICE SHORTCUTS ===

        # Quick stats
        if any(w in text for w in ["stats", "show stats", "summary", "how am i doing"]):
            await self.show_session_stats()
            return

        # Recall last result
        if any(w in text for w in ["last hand", "previous", "what did i do"]):
            await self.recall_last_result()
            return

        # Force tier override - local only
        if any(w in text for w in ["faster", "local only", "offline mode"]):
            self.router.force_local_mode(True)
            await self._speak("Local mode enabled. Cloud APIs disabled.")
            return

        # Force tier override - budget mode
        if any(w in text for w in ["cheaper", "save money", "budget mode"]):
            self.router.prefer_cheap_tier(True)
            await self._speak("Budget mode enabled. Using cheapest models.")
            return

        # Reset mode overrides
        if any(w in text for w in ["full power", "best quality", "no limits"]):
            self.router.force_local_mode(False)
            self.router.prefer_cheap_tier(False)
            await self._speak("Full power mode. All models enabled.")
            return

        # EDITH pause/resume
        if any(w in text for w in ["pause", "stop scanning", "standby"]):
            if self.edith:
                await self.edith.pause()
                await self._speak("EDITH paused. Say resume to continue.")
            return

        if any(w in text for w in ["resume", "start scanning", "wake up"]):
            if self.edith:
                await self.edith.resume()
                await self._speak("EDITH scanning resumed.")
            return

        # === DISPLAY & POWER COMMANDS ===

        # Brightness control
        if "brightness up" in text or "brighter" in text:
            current = self.oled_renderer._brightness
            self.set_brightness(min(1.0, current + 0.2))
            await self._speak(f"Brightness at {self.oled_renderer._brightness:.0%}")
            return

        if "brightness down" in text or "dimmer" in text or "dim" in text:
            current = self.oled_renderer._brightness
            self.set_brightness(max(0.2, current - 0.2))
            await self._speak(f"Brightness at {self.oled_renderer._brightness:.0%}")
            return

        # Power saver mode
        if any(w in text for w in ["power saver", "save power", "low power"]):
            self.power_manager.set_mode_override(PowerMode.SAVER)
            await self._speak("Power saver mode enabled.")
            return

        if any(w in text for w in ["full brightness", "max brightness"]):
            self.power_manager.set_mode_override(PowerMode.FULL)
            await self._speak("Full brightness mode.")
            return

        if "auto power" in text or "auto brightness" in text:
            self.power_manager.set_mode_override(None)
            await self._speak("Auto power management enabled.")
            return

        # Do Not Disturb
        if any(w in text for w in ["do not disturb", "quiet mode", "silence notifications"]):
            self.set_do_not_disturb(True)
            await self._speak("Do Not Disturb enabled. Only critical alerts will show.")
            return

        if any(w in text for w in ["allow notifications", "notifications on", "unquiet"]):
            self.set_do_not_disturb(False)
            await self._speak("Notifications enabled.")
            return

        # Battery status
        if any(w in text for w in ["battery", "power status", "how much power"]):
            status = self.power_manager.get_status_string()
            await self._speak(f"Power status: {status}")
            return

        # === EXISTING COMMANDS ===

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
        """Shutdown the client with session summary."""
        self.running = False
        await self.stop_mode()

        # Generate and display session summary
        if self.config.get('session_summary', {}).get('display_on_shutdown', True):
            try:
                summary = self.session_summary.generate_summary()
                formatted = self.session_summary.format_summary(summary)

                # Display on HUD for 15 seconds
                lua = self.renderer.render_text(formatted, title="SESSION SUMMARY")
                await self._send_display(lua)

                # Print to console
                print("\n" + formatted + "\n")

                # Save to file
                if self.config.get('session_summary', {}).get('save_to_file', True):
                    saved_path = self.session_summary.save_summary(summary)
                    logger.info(f"Session summary saved to {saved_path}")

                # Brief pause to let user see summary
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error generating session summary: {e}")

        await self.disconnect()
        logger.info("WHAM shutdown complete")



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
