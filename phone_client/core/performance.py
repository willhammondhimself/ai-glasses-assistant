"""
Performance Monitor - Track rendering performance and identify bottlenecks.
Provides real-time FPS tracking, frame timing analysis, and performance reports.
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    phase: str = "render"  # render, animation, compose


@dataclass
class PerformanceReport:
    """Summary of performance metrics."""
    avg_fps: float
    min_fps: float
    max_fps: float
    avg_frame_time_ms: float
    min_frame_time_ms: float
    max_frame_time_ms: float
    frame_drops: int
    total_frames: int
    jank_percentage: float  # % of frames over threshold


class PerformanceMonitor:
    """
    Track rendering performance for the Halo display.

    Monitors frame times, calculates FPS, and identifies performance
    bottlenecks like frame drops and jank.
    """

    # Frame time thresholds (ms)
    TARGET_FRAME_TIME = 33.33  # 30 FPS target
    JANK_THRESHOLD = 50.0      # >50ms is noticeable jank
    DROP_THRESHOLD = 100.0     # >100ms is a dropped frame

    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of frames to keep in rolling window
        """
        self.window_size = window_size
        self._frame_times: deque = deque(maxlen=window_size)
        self._animation_times: Dict[str, deque] = {}
        self._phase_times: Dict[str, List[float]] = {}

        self._total_frames = 0
        self._total_frame_drops = 0
        self._session_start = time.time()

        self._callbacks: List[Callable[[FrameMetrics], None]] = []

        logger.debug(f"PerformanceMonitor initialized (window={window_size})")

    def record_frame(self, duration_ms: float, phase: str = "render"):
        """
        Record frame render time.

        Args:
            duration_ms: Frame duration in milliseconds
            phase: Rendering phase (render, animation, compose)
        """
        metrics = FrameMetrics(
            duration_ms=duration_ms,
            phase=phase
        )

        self._frame_times.append(metrics)
        self._total_frames += 1

        # Track frame drops
        if duration_ms > self.DROP_THRESHOLD:
            self._total_frame_drops += 1
            logger.warning(f"Frame drop detected: {duration_ms:.1f}ms")

        # Track by phase
        if phase not in self._phase_times:
            self._phase_times[phase] = []
        self._phase_times[phase].append(duration_ms)
        if len(self._phase_times[phase]) > self.window_size:
            self._phase_times[phase].pop(0)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Performance callback error: {e}")

    def record_animation(self, animation_name: str, duration_ms: float):
        """
        Record animation execution time.

        Args:
            animation_name: Name of the animation
            duration_ms: Duration in milliseconds
        """
        if animation_name not in self._animation_times:
            self._animation_times[animation_name] = deque(maxlen=50)

        self._animation_times[animation_name].append(duration_ms)

    def get_fps(self) -> float:
        """
        Calculate average FPS from recent frames.

        Returns:
            Average frames per second
        """
        if not self._frame_times:
            return 0.0

        avg_frame_time = sum(f.duration_ms for f in self._frame_times) / len(self._frame_times)

        if avg_frame_time <= 0:
            return 0.0

        return 1000.0 / avg_frame_time

    def get_instant_fps(self) -> float:
        """
        Get instantaneous FPS based on last frame.

        Returns:
            Current FPS (may be unstable)
        """
        if not self._frame_times:
            return 0.0

        last_frame_time = self._frame_times[-1].duration_ms
        if last_frame_time <= 0:
            return 0.0

        return 1000.0 / last_frame_time

    def get_report(self) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Returns:
            PerformanceReport with metrics summary
        """
        if not self._frame_times:
            return PerformanceReport(
                avg_fps=0, min_fps=0, max_fps=0,
                avg_frame_time_ms=0, min_frame_time_ms=0, max_frame_time_ms=0,
                frame_drops=0, total_frames=0, jank_percentage=0
            )

        frame_times = [f.duration_ms for f in self._frame_times]

        avg_time = sum(frame_times) / len(frame_times)
        min_time = min(frame_times)
        max_time = max(frame_times)

        # Count janky frames
        jank_frames = len([t for t in frame_times if t > self.JANK_THRESHOLD])
        jank_pct = (jank_frames / len(frame_times)) * 100

        # Recent frame drops (in window)
        recent_drops = len([t for t in frame_times if t > self.DROP_THRESHOLD])

        return PerformanceReport(
            avg_fps=1000.0 / avg_time if avg_time > 0 else 0,
            min_fps=1000.0 / max_time if max_time > 0 else 0,
            max_fps=1000.0 / min_time if min_time > 0 else 0,
            avg_frame_time_ms=avg_time,
            min_frame_time_ms=min_time,
            max_frame_time_ms=max_time,
            frame_drops=recent_drops,
            total_frames=len(frame_times),
            jank_percentage=jank_pct
        )

    def get_animation_stats(self) -> Dict[str, dict]:
        """
        Get timing statistics for each animation type.

        Returns:
            Dict mapping animation names to timing stats
        """
        stats = {}

        for name, times in self._animation_times.items():
            if times:
                times_list = list(times)
                stats[name] = {
                    "avg_ms": sum(times_list) / len(times_list),
                    "min_ms": min(times_list),
                    "max_ms": max(times_list),
                    "count": len(times_list)
                }

        return stats

    def get_phase_stats(self) -> Dict[str, dict]:
        """
        Get timing statistics by rendering phase.

        Returns:
            Dict mapping phase names to timing stats
        """
        stats = {}

        for phase, times in self._phase_times.items():
            if times:
                stats[phase] = {
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "count": len(times)
                }

        return stats

    def get_session_summary(self) -> dict:
        """
        Get summary of entire session performance.

        Returns:
            Dict with session-level metrics
        """
        session_duration = time.time() - self._session_start

        return {
            "session_duration_s": session_duration,
            "total_frames": self._total_frames,
            "total_frame_drops": self._total_frame_drops,
            "drop_rate_pct": (self._total_frame_drops / max(1, self._total_frames)) * 100,
            "avg_fps": self._total_frames / max(1, session_duration)
        }

    def is_performance_good(self) -> bool:
        """
        Quick check if performance is acceptable.

        Returns:
            True if performance meets targets
        """
        report = self.get_report()

        # Check FPS and jank
        return report.avg_fps >= 25 and report.jank_percentage < 10

    def get_performance_grade(self) -> str:
        """
        Get letter grade for current performance.

        Returns:
            Grade: A (excellent) to F (unplayable)
        """
        report = self.get_report()

        if report.avg_fps >= 28 and report.jank_percentage < 5:
            return "A"
        elif report.avg_fps >= 25 and report.jank_percentage < 10:
            return "B"
        elif report.avg_fps >= 20 and report.jank_percentage < 20:
            return "C"
        elif report.avg_fps >= 15:
            return "D"
        else:
            return "F"

    def on_frame_recorded(self, callback: Callable[[FrameMetrics], None]):
        """
        Register callback for frame recordings.

        Args:
            callback: Function(FrameMetrics) called on each frame
        """
        self._callbacks.append(callback)

    def reset(self):
        """Reset all metrics for new session."""
        self._frame_times.clear()
        self._animation_times.clear()
        self._phase_times.clear()
        self._total_frames = 0
        self._total_frame_drops = 0
        self._session_start = time.time()

        logger.info("Performance metrics reset")

    def print_report(self):
        """Print formatted performance report to console."""
        report = self.get_report()
        session = self.get_session_summary()
        grade = self.get_performance_grade()

        print("\n" + "=" * 50)
        print(f"PERFORMANCE REPORT  [Grade: {grade}]")
        print("=" * 50)
        print(f"FPS:        {report.avg_fps:.1f} avg "
              f"({report.min_fps:.1f} min / {report.max_fps:.1f} max)")
        print(f"Frame Time: {report.avg_frame_time_ms:.1f}ms avg "
              f"({report.min_frame_time_ms:.1f} min / {report.max_frame_time_ms:.1f} max)")
        print(f"Jank:       {report.jank_percentage:.1f}% of frames")
        print(f"Drops:      {report.frame_drops} in last {report.total_frames} frames")
        print("-" * 50)
        print(f"Session:    {session['session_duration_s']:.1f}s, "
              f"{session['total_frames']} frames, "
              f"{session['drop_rate_pct']:.2f}% drop rate")
        print("=" * 50 + "\n")


class FrameTimer:
    """
    Context manager for timing frame rendering.

    Usage:
        monitor = PerformanceMonitor()
        with FrameTimer(monitor, "render"):
            # render code here
            pass
    """

    def __init__(self, monitor: PerformanceMonitor, phase: str = "render"):
        self.monitor = monitor
        self.phase = phase
        self.start_time: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.monitor.record_frame(duration_ms, self.phase)
        return False


# Test
def test_performance_monitor():
    """Test performance monitor functionality."""
    import random

    print("=== Performance Monitor Test ===\n")

    monitor = PerformanceMonitor(window_size=50)

    # Simulate frames
    print("Simulating 100 frames...")
    for i in range(100):
        # Simulate varying frame times (mostly good, some slow)
        if random.random() < 0.1:  # 10% slow frames
            duration = random.uniform(40, 80)
        elif random.random() < 0.02:  # 2% dropped frames
            duration = random.uniform(100, 200)
        else:
            duration = random.uniform(20, 35)

        monitor.record_frame(duration)

        # Simulate some animations
        if random.random() < 0.3:
            monitor.record_animation("fade", random.uniform(5, 15))
        if random.random() < 0.2:
            monitor.record_animation("slide", random.uniform(10, 25))

    # Print report
    monitor.print_report()

    # Animation stats
    print("Animation Stats:")
    for name, stats in monitor.get_animation_stats().items():
        print(f"  {name}: {stats['avg_ms']:.1f}ms avg ({stats['count']} samples)")

    # Test FrameTimer
    print("\nFrameTimer test:")
    with FrameTimer(monitor, "test_phase"):
        time.sleep(0.025)  # 25ms

    print(f"After timer: {monitor.get_instant_fps():.1f} FPS")


if __name__ == "__main__":
    test_performance_monitor()
