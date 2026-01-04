"""
Animations - Smooth transitions and visual effects for Halo display.
Generates timed Lua sequences for professional-grade animations.
"""
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Callable, Optional

from .oled_renderer import OLEDRenderer, OLEDColors, rgb_to_lua, lerp_color


class AnimationType(Enum):
    """Available animation types."""
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    PULSE = "pulse"
    SCALE = "scale"
    TYPEWRITER = "typewriter"
    NUMBER_ROLL = "number_roll"


class AnimationTimings:
    """
    Scientifically calibrated animation timings based on Google Material Design
    motion guidelines and UX research for optimal user experience.

    Reference: https://m3.material.io/styles/motion/overview

    Human perception thresholds:
    - <100ms: Perceived as instant
    - 100-300ms: Perceived as quick, responsive
    - 300-500ms: Smooth, comfortable transitions
    - 500-800ms: Deliberate, attention-grabbing
    - >800ms: Slow, may feel sluggish
    """

    # Base timings (milliseconds)
    INSTANT = 50           # 0.05s - Immediate feedback (tap response)
    QUICK = 150            # 0.15s - Button press, quick feedback
    STANDARD = 300         # 0.30s - Default transitions, mode changes
    SMOOTH = 500           # 0.50s - Complex transitions, reveals
    DRAMATIC = 800         # 0.80s - Attention-grabbing, celebrations

    # Context-specific timings (optimized for each use case)

    # Notifications
    NOTIFICATION_IN = 300      # Enter animation - noticeable but not jarring
    NOTIFICATION_OUT = 200     # Exit animation - faster than enter (UX principle)
    NOTIFICATION_AUTO_DISMISS = 5000  # Default auto-dismiss time

    # Mode transitions
    MODE_SWITCH = 400          # Switching between poker/math/meeting modes
    THEME_TRANSITION = 300     # Color theme changes

    # Feedback animations
    ERROR_FLASH = 150          # Fast attention grab for errors
    SUCCESS_PULSE = 300        # Celebratory but not distracting
    WARNING_PULSE = 400        # Slower to ensure visibility

    # Loading/thinking states
    THINKING_PULSE = 800       # Slow pulse while AI is processing
    LOADING_DOT = 400          # Dot animation in loading indicator

    # Poker-specific
    CARD_REVEAL = 200          # Quick card reveal
    RECOMMENDATION_IN = 400    # Recommendation appearing
    CONFIDENCE_FILL = 500      # Confidence bar filling

    # Mental math specific
    PROBLEM_IN = 300           # New problem appearing
    RESULT_REVEAL = 250        # Answer reveal
    STREAK_CELEBRATION = 600   # Streak milestone celebration
    TIMER_PULSE = 300          # Timer warning pulse

    # Typewriter effects
    CHAR_DELAY_FAST = 30       # Fast typewriter (WHAM-style)
    CHAR_DELAY_NORMAL = 50     # Normal typewriter
    CHAR_DELAY_DRAMATIC = 80   # Dramatic reveal

    # Number animations
    NUMBER_ROLL_FAST = 300     # Quick number change
    NUMBER_ROLL_NORMAL = 500   # Standard number animation
    NUMBER_ROLL_SLOW = 800     # Dramatic count

    # Progress indicators
    PROGRESS_UPDATE = 200      # Progress bar updates
    PROGRESS_COMPLETE = 400    # Completion animation

    @classmethod
    def get_timing(cls, context: str) -> int:
        """
        Get recommended timing for a specific context.

        Args:
            context: Animation context string

        Returns:
            Timing in milliseconds
        """
        context_map = {
            # General
            "instant": cls.INSTANT,
            "quick": cls.QUICK,
            "standard": cls.STANDARD,
            "smooth": cls.SMOOTH,
            "dramatic": cls.DRAMATIC,

            # Notifications
            "notification_in": cls.NOTIFICATION_IN,
            "notification_out": cls.NOTIFICATION_OUT,

            # Mode changes
            "mode_switch": cls.MODE_SWITCH,
            "theme_change": cls.THEME_TRANSITION,

            # Feedback
            "error": cls.ERROR_FLASH,
            "success": cls.SUCCESS_PULSE,
            "warning": cls.WARNING_PULSE,
            "thinking": cls.THINKING_PULSE,

            # Poker
            "card_reveal": cls.CARD_REVEAL,
            "recommendation": cls.RECOMMENDATION_IN,
            "confidence": cls.CONFIDENCE_FILL,

            # Math
            "problem": cls.PROBLEM_IN,
            "result": cls.RESULT_REVEAL,
            "streak": cls.STREAK_CELEBRATION,

            # Default
            "default": cls.STANDARD,
        }
        return context_map.get(context, cls.STANDARD)

    @classmethod
    def apply_multiplier(cls, timing: int, multiplier: float) -> int:
        """
        Apply a multiplier to a timing value.

        Args:
            timing: Base timing in milliseconds
            multiplier: Speed multiplier (0.5 = faster, 2.0 = slower)

        Returns:
            Adjusted timing in milliseconds
        """
        return max(cls.INSTANT, int(timing * multiplier))


class Easing(Enum):
    """Easing function types."""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    BOUNCE = "bounce"


@dataclass
class Animation:
    """Animation configuration."""
    type: AnimationType
    duration_ms: int = 300
    easing: Easing = Easing.EASE_OUT
    delay_ms: int = 0


def ease_linear(t: float) -> float:
    """Linear easing (no acceleration)."""
    return t


def ease_in(t: float) -> float:
    """Ease-in (accelerate from zero)."""
    return t * t


def ease_out(t: float) -> float:
    """Ease-out (decelerate to zero) - recommended for UI."""
    return 1 - (1 - t) * (1 - t)


def ease_in_out(t: float) -> float:
    """Ease-in-out (smooth both ends)."""
    if t < 0.5:
        return 2 * t * t
    return 1 - pow(-2 * t + 2, 2) / 2


def ease_bounce(t: float) -> float:
    """Bounce easing for playful effects."""
    n1 = 7.5625
    d1 = 2.75

    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


def get_easing_func(easing: Easing) -> Callable[[float], float]:
    """Get easing function by type."""
    easing_map = {
        Easing.LINEAR: ease_linear,
        Easing.EASE_IN: ease_in,
        Easing.EASE_OUT: ease_out,
        Easing.EASE_IN_OUT: ease_in_out,
        Easing.BOUNCE: ease_bounce,
    }
    return easing_map.get(easing, ease_out)


def interpolate(start: float, end: float, t: float, easing: Easing = Easing.EASE_OUT) -> float:
    """
    Interpolate between two values with easing.

    Args:
        start: Start value
        end: End value
        t: Progress (0.0-1.0)
        easing: Easing function type

    Returns:
        Interpolated value
    """
    ease_func = get_easing_func(easing)
    eased_t = ease_func(max(0.0, min(1.0, t)))
    return start + (end - start) * eased_t


class AnimationEngine:
    """
    Generate smooth animation Lua sequences for Halo display.

    Animations are generated as a series of Lua commands with timing
    that can be sent to the glasses in sequence.
    """

    # Target frame rate for animations
    TARGET_FPS = 30
    FRAME_MS = 1000 / TARGET_FPS  # ~33ms per frame

    def __init__(
        self,
        renderer: OLEDRenderer = None,
        enabled: bool = True,
        duration_multiplier: float = 1.0,
        reduce_motion: bool = False
    ):
        """
        Initialize animation engine.

        Args:
            renderer: OLED renderer instance
            enabled: Whether animations are enabled
            duration_multiplier: Scale animation durations (0.5 = faster, 2.0 = slower)
            reduce_motion: Accessibility option to minimize motion
        """
        self.renderer = renderer or OLEDRenderer()
        self.enabled = enabled
        self.duration_multiplier = duration_multiplier
        self.reduce_motion = reduce_motion

    def _get_frame_count(self, duration_ms: int) -> int:
        """Calculate number of frames for duration."""
        adjusted_ms = duration_ms * self.duration_multiplier
        return max(1, int(adjusted_ms / self.FRAME_MS))

    def _generate_frames(
        self,
        frame_count: int,
        easing: Easing,
        frame_callback: Callable[[float, int], str]
    ) -> List[Tuple[str, int]]:
        """
        Generate animation frames.

        Args:
            frame_count: Number of frames
            easing: Easing function
            frame_callback: Function(progress, frame_index) -> Lua code

        Returns:
            List of (lua_code, delay_ms) tuples
        """
        frames = []
        ease_func = get_easing_func(easing)

        for i in range(frame_count):
            t = i / max(1, frame_count - 1)
            eased_t = ease_func(t)

            lua = frame_callback(eased_t, i)
            delay = int(self.FRAME_MS)

            frames.append((lua, delay))

        return frames

    def fade_transition(
        self,
        from_content: Callable[[OLEDRenderer], None],
        to_content: Callable[[OLEDRenderer], None],
        duration_ms: int = 300,
        easing: Easing = Easing.EASE_OUT
    ) -> List[Tuple[str, int]]:
        """
        Generate crossfade transition between two screens.

        Args:
            from_content: Function that renders the "from" screen
            to_content: Function that renders the "to" screen
            duration_ms: Animation duration
            easing: Easing function

        Returns:
            List of (lua_code, delay_ms) frame tuples
        """
        if self.reduce_motion or not self.enabled:
            # Instant transition
            self.renderer.clear()
            to_content(self.renderer)
            self.renderer.show()
            return [(self.renderer.get_lua(), 0)]

        frame_count = self._get_frame_count(duration_ms)

        def render_frame(progress: float, frame_idx: int) -> str:
            # For now, we simulate fade by showing the destination
            # True crossfade would require alpha blending which Halo may not support
            self.renderer.clear()
            to_content(self.renderer)
            self.renderer.show()
            return self.renderer.get_lua()

        return self._generate_frames(frame_count, easing, render_frame)

    def slide_in(
        self,
        content: Callable[[OLEDRenderer, int], None],
        direction: str = "left",
        duration_ms: int = 300,
        easing: Easing = Easing.EASE_OUT
    ) -> List[Tuple[str, int]]:
        """
        Slide content in from edge.

        Args:
            content: Function(renderer, x_offset) that renders content
            direction: Direction to slide from (left, right, up, down)
            duration_ms: Animation duration
            easing: Easing function

        Returns:
            List of (lua_code, delay_ms) frame tuples
        """
        if self.reduce_motion or not self.enabled:
            self.renderer.clear()
            content(self.renderer, 0)
            self.renderer.show()
            return [(self.renderer.get_lua(), 0)]

        frame_count = self._get_frame_count(duration_ms)

        # Determine start offset based on direction
        offsets = {
            "left": (-self.renderer.width, 0),
            "right": (self.renderer.width, 0),
            "up": (0, -self.renderer.height),
            "down": (0, self.renderer.height),
        }
        start_x, start_y = offsets.get(direction, (-self.renderer.width, 0))

        def render_frame(progress: float, frame_idx: int) -> str:
            x_offset = int(interpolate(start_x, 0, progress, easing))
            y_offset = int(interpolate(start_y, 0, progress, easing))

            self.renderer.clear()
            # For slide, we'd need to offset all content
            # Simplified: just render at final position after halfway
            if progress > 0.5:
                content(self.renderer, 0)
            self.renderer.show()
            return self.renderer.get_lua()

        return self._generate_frames(frame_count, easing, render_frame)

    def pulse_element(
        self,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int] = OLEDColors.PRIMARY,
        cycles: int = 2,
        duration_ms: int = 800
    ) -> List[Tuple[str, int]]:
        """
        Create pulsing effect on an element.

        Args:
            x: X position
            y: Y position
            text: Text content
            color: Pulse color
            cycles: Number of pulse cycles
            duration_ms: Total duration

        Returns:
            List of (lua_code, delay_ms) frame tuples
        """
        if self.reduce_motion or not self.enabled:
            self.renderer.clear()
            self.renderer.text(text, x, y, color)
            self.renderer.show()
            return [(self.renderer.get_lua(), 0)]

        frame_count = self._get_frame_count(duration_ms)
        frames = []

        for i in range(frame_count):
            progress = i / max(1, frame_count - 1)

            # Sine wave for smooth pulsing
            pulse = (math.sin(progress * cycles * 2 * math.pi) + 1) / 2

            # Interpolate color brightness
            dim_color = (color[0] // 3, color[1] // 3, color[2] // 3)
            current_color = lerp_color(dim_color, color, pulse)

            self.renderer.clear()
            self.renderer.text(text, x, y, current_color)
            self.renderer.show()

            frames.append((self.renderer.get_lua(), int(self.FRAME_MS)))

        return frames

    def typewriter_effect(
        self,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int] = OLEDColors.TEXT_PRIMARY,
        char_delay_ms: int = 50
    ) -> List[Tuple[str, int]]:
        """
        Character-by-character reveal effect.

        Args:
            x: X position
            y: Y position
            text: Text to reveal
            color: Text color
            char_delay_ms: Delay per character

        Returns:
            List of (lua_code, delay_ms) frame tuples
        """
        if self.reduce_motion or not self.enabled:
            self.renderer.clear()
            self.renderer.text(text, x, y, color)
            self.renderer.show()
            return [(self.renderer.get_lua(), 0)]

        frames = []

        for i in range(len(text) + 1):
            partial = text[:i]

            self.renderer.clear()
            self.renderer.text(partial + "_", x, y, color)  # Cursor effect
            self.renderer.show()

            delay = int(char_delay_ms * self.duration_multiplier)
            frames.append((self.renderer.get_lua(), delay))

        # Final frame without cursor
        self.renderer.clear()
        self.renderer.text(text, x, y, color)
        self.renderer.show()
        frames.append((self.renderer.get_lua(), 0))

        return frames

    def number_roll(
        self,
        x: int,
        y: int,
        from_value: float,
        to_value: float,
        format_str: str = "{:.0f}",
        color: Tuple[int, int, int] = OLEDColors.TEXT_PRIMARY,
        duration_ms: int = 500,
        easing: Easing = Easing.EASE_OUT
    ) -> List[Tuple[str, int]]:
        """
        Animate a number changing.

        Args:
            x: X position
            y: Y position
            from_value: Starting value
            to_value: Ending value
            format_str: Format string for number
            color: Text color
            duration_ms: Animation duration
            easing: Easing function

        Returns:
            List of (lua_code, delay_ms) frame tuples
        """
        if self.reduce_motion or not self.enabled:
            self.renderer.clear()
            self.renderer.text(format_str.format(to_value), x, y, color)
            self.renderer.show()
            return [(self.renderer.get_lua(), 0)]

        frame_count = self._get_frame_count(duration_ms)

        def render_frame(progress: float, frame_idx: int) -> str:
            current = interpolate(from_value, to_value, progress, easing)

            self.renderer.clear()
            self.renderer.text(format_str.format(current), x, y, color)
            self.renderer.show()

            return self.renderer.get_lua()

        return self._generate_frames(frame_count, easing, render_frame)

    def progress_fill(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        from_value: float,
        to_value: float,
        color: Tuple[int, int, int] = OLEDColors.PRIMARY,
        duration_ms: int = 500,
        easing: Easing = Easing.EASE_OUT
    ) -> List[Tuple[str, int]]:
        """
        Animate a progress bar filling.

        Args:
            x: X position
            y: Y position
            width: Bar width
            height: Bar height
            from_value: Starting value (0.0-1.0)
            to_value: Ending value (0.0-1.0)
            color: Bar color
            duration_ms: Animation duration
            easing: Easing function

        Returns:
            List of (lua_code, delay_ms) frame tuples
        """
        if self.reduce_motion or not self.enabled:
            self.renderer.clear()
            self.renderer.progress_bar(x, y, width, height, to_value, 1.0, color)
            self.renderer.show()
            return [(self.renderer.get_lua(), 0)]

        frame_count = self._get_frame_count(duration_ms)

        def render_frame(progress: float, frame_idx: int) -> str:
            current = interpolate(from_value, to_value, progress, easing)

            self.renderer.clear()
            self.renderer.progress_bar(x, y, width, height, current, 1.0, color)
            self.renderer.show()

            return self.renderer.get_lua()

        return self._generate_frames(frame_count, easing, render_frame)

    def attention_flash(
        self,
        color: Tuple[int, int, int] = OLEDColors.WARNING,
        flashes: int = 2,
        duration_ms: int = 400
    ) -> List[Tuple[str, int]]:
        """
        Flash screen border for attention.

        Args:
            color: Flash color
            flashes: Number of flashes
            duration_ms: Total duration

        Returns:
            List of (lua_code, delay_ms) frame tuples
        """
        if self.reduce_motion or not self.enabled:
            return []

        frames = []
        flash_duration = duration_ms // (flashes * 2)

        for _ in range(flashes):
            # Flash on
            self.renderer.clear()
            self.renderer.rect(0, 0, self.renderer.width, 4, color, filled=True)  # Top
            self.renderer.rect(0, self.renderer.height - 4, self.renderer.width, 4, color, filled=True)  # Bottom
            self.renderer.rect(0, 0, 4, self.renderer.height, color, filled=True)  # Left
            self.renderer.rect(self.renderer.width - 4, 0, 4, self.renderer.height, color, filled=True)  # Right
            self.renderer.show()
            frames.append((self.renderer.get_lua(), flash_duration))

            # Flash off
            self.renderer.clear()
            self.renderer.show()
            frames.append((self.renderer.get_lua(), flash_duration))

        return frames

    def create_sequence(self, animations: List[Tuple[str, int]]) -> str:
        """
        Create a single Lua script with timing for animation sequence.

        Args:
            animations: List of (lua_code, delay_ms) tuples

        Returns:
            Combined Lua script with sleep commands
        """
        parts = []

        for lua, delay in animations:
            parts.append(lua)
            if delay > 0:
                parts.append(f"frame.sleep({delay})")

        return "\n".join(parts)


# Factory function
def create_animation_engine(
    config: dict = None,
    renderer: OLEDRenderer = None
) -> AnimationEngine:
    """
    Factory function to create an animation engine.

    Args:
        config: Configuration dict with 'animations' section
        renderer: Optional OLED renderer

    Returns:
        Configured AnimationEngine instance
    """
    enabled = True
    duration_multiplier = 1.0
    reduce_motion = False

    if config and "animations" in config:
        anim_config = config["animations"]
        enabled = anim_config.get("enabled", True)
        duration_multiplier = anim_config.get("duration_multiplier", 1.0)
        reduce_motion = anim_config.get("reduce_motion", False)

    return AnimationEngine(
        renderer=renderer,
        enabled=enabled,
        duration_multiplier=duration_multiplier,
        reduce_motion=reduce_motion
    )


# Test
def test_animations():
    """Test animation engine."""
    print("=== Animation Engine Test ===\n")

    engine = AnimationEngine()

    # Test typewriter
    print("Typewriter effect:")
    frames = engine.typewriter_effect(320, 200, "Hello WHAM", OLEDColors.PRIMARY, 80)
    print(f"  Generated {len(frames)} frames")
    print(f"  First frame preview: {frames[0][0][:100]}...")
    print()

    # Test pulse
    print("Pulse effect:")
    frames = engine.pulse_element(320, 200, "RAISE", OLEDColors.WARNING, 3, 600)
    print(f"  Generated {len(frames)} frames")
    print()

    # Test number roll
    print("Number roll:")
    frames = engine.number_roll(320, 200, 0, 100, "{:.0f}%", OLEDColors.SUCCESS, 500)
    print(f"  Generated {len(frames)} frames")
    print()

    # Test progress fill
    print("Progress fill:")
    frames = engine.progress_fill(100, 200, 400, 20, 0.0, 0.85, OLEDColors.PRIMARY, 600)
    print(f"  Generated {len(frames)} frames")
    print()

    # Test attention flash
    print("Attention flash:")
    frames = engine.attention_flash(OLEDColors.ERROR, 3, 600)
    print(f"  Generated {len(frames)} frames")
    print()

    # Test with reduce_motion
    print("With reduce_motion enabled:")
    engine_accessible = AnimationEngine(reduce_motion=True)
    frames = engine_accessible.typewriter_effect(320, 200, "Instant", OLEDColors.PRIMARY)
    print(f"  Generated {len(frames)} frames (instant)")


if __name__ == "__main__":
    test_animations()
