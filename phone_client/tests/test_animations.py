"""
Animation Engine Tests.
Tests easing functions, frame generation, and animation timing.
"""
import pytest
import math

# Path setup via conftest.py


from halo.animations import (
    AnimationEngine, AnimationType, Easing, Animation,
    ease_linear, ease_in, ease_out, ease_in_out, ease_bounce,
    get_easing_func, interpolate
)
from halo.oled_renderer import OLEDRenderer, OLEDColors


class TestEasingFunctions:
    """Test easing function implementations."""

    def test_ease_linear(self):
        """Test linear easing (no acceleration)."""
        assert ease_linear(0.0) == 0.0
        assert ease_linear(0.5) == 0.5
        assert ease_linear(1.0) == 1.0

    def test_ease_in(self):
        """Test ease-in (accelerate from zero)."""
        # Starts slow
        assert ease_in(0.0) == 0.0
        assert ease_in(0.25) < 0.25
        # Ends at 1
        assert ease_in(1.0) == 1.0
        # Middle should be less than linear
        assert ease_in(0.5) == 0.25  # t^2

    def test_ease_out(self):
        """Test ease-out (decelerate to zero)."""
        # Starts fast
        assert ease_out(0.0) == 0.0
        assert ease_out(0.25) > 0.25
        # Ends at 1
        assert ease_out(1.0) == 1.0
        # Formula: 1 - (1-t)^2 at 0.5 = 1 - 0.25 = 0.75
        assert ease_out(0.5) == 0.75

    def test_ease_in_out(self):
        """Test ease-in-out (smooth both ends)."""
        # Start and end
        assert ease_in_out(0.0) == 0.0
        assert ease_in_out(1.0) == 1.0
        # Middle is exactly 0.5
        assert ease_in_out(0.5) == 0.5
        # First half is ease-in-like (below linear)
        assert ease_in_out(0.25) < 0.25
        # Second half is ease-out-like (above linear)
        assert ease_in_out(0.75) > 0.75

    def test_ease_bounce(self):
        """Test bounce easing for playful effects."""
        # Start and end
        assert ease_bounce(0.0) == 0.0
        assert abs(ease_bounce(1.0) - 1.0) < 0.01
        # Bouncy behavior - value increases non-linearly
        mid = ease_bounce(0.5)
        assert mid > 0 and mid < 1

    def test_get_easing_func(self):
        """Test easing function lookup."""
        assert get_easing_func(Easing.LINEAR) == ease_linear
        assert get_easing_func(Easing.EASE_IN) == ease_in
        assert get_easing_func(Easing.EASE_OUT) == ease_out
        assert get_easing_func(Easing.EASE_IN_OUT) == ease_in_out
        assert get_easing_func(Easing.BOUNCE) == ease_bounce


class TestInterpolation:
    """Test value interpolation with easing."""

    def test_interpolate_basic(self):
        """Test basic interpolation."""
        # Linear from 0 to 100
        assert interpolate(0, 100, 0.0) == 0
        assert interpolate(0, 100, 0.5) == 75  # ease_out default
        assert interpolate(0, 100, 1.0) == 100

    def test_interpolate_linear(self):
        """Test linear interpolation."""
        assert interpolate(0, 100, 0.0, Easing.LINEAR) == 0
        assert interpolate(0, 100, 0.5, Easing.LINEAR) == 50
        assert interpolate(0, 100, 1.0, Easing.LINEAR) == 100

    def test_interpolate_negative(self):
        """Test interpolation with negative values."""
        assert interpolate(-50, 50, 0.5, Easing.LINEAR) == 0
        assert interpolate(100, -100, 0.5, Easing.LINEAR) == 0

    def test_interpolate_clamping(self):
        """Test t value is clamped."""
        # Over 1.0
        assert interpolate(0, 100, 1.5, Easing.LINEAR) == 100
        # Under 0.0
        assert interpolate(0, 100, -0.5, Easing.LINEAR) == 0


class TestAnimationFrameGeneration:
    """Test animation frame generation."""

    def test_frame_count_calculation(self):
        """Test frame count based on duration."""
        engine = AnimationEngine()

        # 30 FPS target, 33ms per frame
        # 300ms duration = ~9 frames
        frames = engine._get_frame_count(300)
        assert frames == 9

        # 1000ms = ~30 frames
        frames = engine._get_frame_count(1000)
        assert frames == 30

    def test_frame_count_with_multiplier(self):
        """Test frame count with duration multiplier."""
        # 0.5x speed (faster)
        engine = AnimationEngine(duration_multiplier=0.5)
        frames = engine._get_frame_count(300)
        assert frames == 4  # Half the frames

        # 2x speed (slower)
        engine = AnimationEngine(duration_multiplier=2.0)
        frames = engine._get_frame_count(300)
        assert frames == 18  # Double the frames

    def test_minimum_one_frame(self):
        """Test minimum frame count is 1."""
        engine = AnimationEngine()
        frames = engine._get_frame_count(10)  # Very short
        assert frames >= 1

    def test_animation_generates_at_30fps(self):
        """Verify animations generate expected frame counts at 30fps."""
        engine = AnimationEngine()

        # 1 second animation at 30fps = 30 frames
        frames = engine.typewriter_effect(100, 100, "Hello", char_delay_ms=33)
        # 5 characters + 1 final = 6 frames for typewriter
        assert len(frames) == 6

        # Pulse with 800ms
        frames = engine.pulse_element(100, 100, "Test", duration_ms=800)
        expected = engine._get_frame_count(800)
        assert len(frames) == expected


class TestAnimationEffects:
    """Test specific animation effects."""

    def test_typewriter_effect(self):
        """Test character-by-character reveal."""
        engine = AnimationEngine()
        frames = engine.typewriter_effect(100, 100, "Hello", char_delay_ms=50)

        # 5 chars + final = 6 frames
        assert len(frames) == 6

        # Each frame is (lua_code, delay_ms)
        for lua, delay in frames[:-1]:  # All except last
            assert "frame.display" in lua
            assert delay == 50

        # Last frame has no delay
        assert frames[-1][1] == 0

    def test_pulse_element(self):
        """Test pulsing effect."""
        engine = AnimationEngine()
        frames = engine.pulse_element(100, 100, "Pulse", cycles=2, duration_ms=600)

        assert len(frames) > 0
        for lua, delay in frames:
            assert "frame.display" in lua
            assert "Pulse" in lua

    def test_number_roll(self):
        """Test number animation."""
        engine = AnimationEngine()
        frames = engine.number_roll(
            100, 100,
            from_value=0,
            to_value=100,
            format_str="{:.0f}",
            duration_ms=300
        )

        assert len(frames) > 0
        # First frame should have 0 or close
        # Last frame should have 100
        assert "100" in frames[-1][0]

    def test_progress_fill(self):
        """Test progress bar animation."""
        engine = AnimationEngine()
        frames = engine.progress_fill(
            100, 100, 200, 20,
            from_value=0.0,
            to_value=0.8,
            duration_ms=500
        )

        assert len(frames) > 0
        for lua, delay in frames:
            assert "frame.display" in lua

    def test_attention_flash(self):
        """Test attention flash effect."""
        engine = AnimationEngine()
        frames = engine.attention_flash(flashes=3, duration_ms=600)

        # 3 flashes = 6 frames (on/off each)
        assert len(frames) == 6


class TestReducedMotion:
    """Test accessibility reduced motion option."""

    def test_reduce_motion_typewriter(self):
        """Test typewriter is instant with reduce_motion."""
        engine = AnimationEngine(reduce_motion=True)
        frames = engine.typewriter_effect(100, 100, "Hello World")

        # Should be single instant frame
        assert len(frames) == 1
        assert frames[0][1] == 0  # No delay

    def test_reduce_motion_pulse(self):
        """Test pulse is instant with reduce_motion."""
        engine = AnimationEngine(reduce_motion=True)
        frames = engine.pulse_element(100, 100, "Test", cycles=5)

        assert len(frames) == 1
        assert frames[0][1] == 0

    def test_reduce_motion_number_roll(self):
        """Test number roll shows final value with reduce_motion."""
        engine = AnimationEngine(reduce_motion=True)
        frames = engine.number_roll(100, 100, 0, 50, "{:.0f}")

        assert len(frames) == 1
        assert "50" in frames[0][0]

    def test_reduce_motion_attention_flash(self):
        """Test attention flash is empty with reduce_motion."""
        engine = AnimationEngine(reduce_motion=True)
        frames = engine.attention_flash(flashes=5)

        assert len(frames) == 0


class TestAnimationDisabled:
    """Test behavior when animations are disabled."""

    def test_disabled_typewriter(self):
        """Test typewriter shows final text when disabled."""
        engine = AnimationEngine(enabled=False)
        frames = engine.typewriter_effect(100, 100, "Hello")

        assert len(frames) == 1
        assert "Hello" in frames[0][0]

    def test_disabled_fade(self):
        """Test fade shows destination when disabled."""
        engine = AnimationEngine(enabled=False)

        def from_content(r):
            r.text("From", 100, 100)

        def to_content(r):
            r.text("To", 100, 100)

        frames = engine.fade_transition(from_content, to_content)

        assert len(frames) == 1
        assert "To" in frames[0][0]


class TestSlideAnimation:
    """Test slide animation directions."""

    def test_slide_in_left(self):
        """Test slide in from left."""
        engine = AnimationEngine()

        def content(r, offset):
            r.text("Content", 100 + offset, 100)

        frames = engine.slide_in(content, "left", duration_ms=300)
        assert len(frames) > 0

    def test_slide_in_right(self):
        """Test slide in from right."""
        engine = AnimationEngine()

        def content(r, offset):
            r.text("Content", 100, 100)

        frames = engine.slide_in(content, "right", duration_ms=300)
        assert len(frames) > 0

    def test_slide_in_up(self):
        """Test slide in from top."""
        engine = AnimationEngine()

        def content(r, offset):
            r.text("Content", 100, 100)

        frames = engine.slide_in(content, "up", duration_ms=300)
        assert len(frames) > 0

    def test_slide_in_down(self):
        """Test slide in from bottom."""
        engine = AnimationEngine()

        def content(r, offset):
            r.text("Content", 100, 100)

        frames = engine.slide_in(content, "down", duration_ms=300)
        assert len(frames) > 0


class TestSequenceCreation:
    """Test animation sequence creation."""

    def test_create_sequence(self):
        """Test creating Lua script from animation frames."""
        engine = AnimationEngine()

        frames = [
            ("frame.display.text('1')", 100),
            ("frame.display.text('2')", 100),
            ("frame.display.text('3')", 0),  # Last frame no delay
        ]

        sequence = engine.create_sequence(frames)

        assert "frame.display.text('1')" in sequence
        assert "frame.sleep(100)" in sequence
        assert "frame.display.text('3')" in sequence

    def test_create_sequence_no_delay(self):
        """Test sequence with zero delays."""
        engine = AnimationEngine()

        frames = [
            ("frame.display.clear()", 0),
            ("frame.display.show()", 0),
        ]

        sequence = engine.create_sequence(frames)

        # Should not contain sleep commands
        assert "frame.sleep" not in sequence


class TestAnimationConfiguration:
    """Test animation configuration dataclass."""

    def test_animation_defaults(self):
        """Test Animation dataclass defaults."""
        anim = Animation(type=AnimationType.FADE_IN)

        assert anim.type == AnimationType.FADE_IN
        assert anim.duration_ms == 300
        assert anim.easing == Easing.EASE_OUT
        assert anim.delay_ms == 0

    def test_animation_custom(self):
        """Test custom Animation configuration."""
        anim = Animation(
            type=AnimationType.SLIDE_LEFT,
            duration_ms=500,
            easing=Easing.BOUNCE,
            delay_ms=100
        )

        assert anim.duration_ms == 500
        assert anim.easing == Easing.BOUNCE
        assert anim.delay_ms == 100


class TestAnimationTypes:
    """Test animation type enumeration."""

    def test_animation_types_exist(self):
        """Verify all animation types are defined."""
        expected_types = [
            "FADE_IN", "FADE_OUT",
            "SLIDE_LEFT", "SLIDE_RIGHT", "SLIDE_UP", "SLIDE_DOWN",
            "PULSE", "SCALE", "TYPEWRITER", "NUMBER_ROLL"
        ]

        for type_name in expected_types:
            assert hasattr(AnimationType, type_name)


class TestEngineWithRenderer:
    """Test animation engine integration with renderer."""

    def test_engine_creates_renderer(self):
        """Test engine creates default renderer if not provided."""
        engine = AnimationEngine()
        assert engine.renderer is not None
        assert isinstance(engine.renderer, OLEDRenderer)

    def test_engine_uses_provided_renderer(self):
        """Test engine uses provided renderer."""
        renderer = OLEDRenderer()
        engine = AnimationEngine(renderer=renderer)
        assert engine.renderer is renderer


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
