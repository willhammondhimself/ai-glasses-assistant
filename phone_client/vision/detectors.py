"""
EDITH Content Detectors.
OpenCV-based lightweight detection for equations, code, and text.
"""
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Lazy import OpenCV
cv2 = None
np = None


def _load_cv2():
    """Lazy load OpenCV."""
    global cv2, np
    if cv2 is None:
        try:
            import cv2 as _cv2
            import numpy as _np
            cv2 = _cv2
            np = _np
        except ImportError:
            logger.warning("opencv-python not installed. Run: pip install opencv-python")
    return cv2 is not None


@dataclass
class DetectionResult:
    """Result from a content detector."""
    detected: bool
    content_type: str
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    region_image: Optional[bytes] = None  # Cropped region for OCR


class ContentDetector(ABC):
    """Base class for content detectors."""

    @abstractmethod
    def detect(self, frame) -> Optional[DetectionResult]:
        """Detect content in frame."""
        pass


class EquationDetector(ContentDetector):
    """
    Detect mathematical equations via OpenCV patterns.

    Detection strategy:
    1. Edge detection for text regions
    2. Look for horizontal lines (fraction bars, equals signs)
    3. Look for math symbol density
    4. Check for structured formula layout
    """

    # Minimum percentage of frame that must contain edges for detection
    MIN_EDGE_RATIO = 0.02
    MAX_EDGE_RATIO = 0.30

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self._cv2_available = _load_cv2()

    def detect(self, frame) -> Optional[DetectionResult]:
        """Detect equation in frame."""
        if not self._cv2_available:
            return None

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Calculate edge density
            edge_ratio = np.sum(edges > 0) / edges.size

            if edge_ratio < self.MIN_EDGE_RATIO or edge_ratio > self.MAX_EDGE_RATIO:
                return None

            # Look for horizontal lines (fraction bars, equals signs)
            horizontal_score = self._detect_horizontal_lines(edges)

            # Look for math-like structure (centered, clean background)
            structure_score = self._detect_structured_layout(gray, edges)

            # Combined confidence
            confidence = (horizontal_score * 0.4 + structure_score * 0.4 + edge_ratio * 0.2)

            if confidence >= self.min_confidence:
                # Find bounding box of equation region
                bbox = self._find_content_bbox(edges)

                # Crop region for OCR
                region = self._crop_region(frame, bbox) if bbox else None

                return DetectionResult(
                    detected=True,
                    content_type="equation",
                    confidence=confidence,
                    bounding_box=bbox,
                    region_image=region
                )

            return None

        except Exception as e:
            logger.debug(f"Equation detection error: {e}")
            return None

    def _detect_horizontal_lines(self, edges) -> float:
        """Detect horizontal lines (fraction bars, equals signs)."""
        # Use Hough transform to find lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )

        if lines is None:
            return 0.0

        horizontal_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15 or angle > 165:  # Nearly horizontal
                horizontal_count += 1

        # Normalize by image height
        return min(1.0, horizontal_count / 10)

    def _detect_structured_layout(self, gray, edges) -> float:
        """Detect structured math-like layout."""
        h, w = gray.shape

        # Check for clean background (low variance in non-edge areas)
        mask = edges == 0
        if np.sum(mask) > 0:
            bg_std = np.std(gray[mask])
            bg_score = 1.0 - min(1.0, bg_std / 50)
        else:
            bg_score = 0

        # Check for centered content
        center_region = edges[h//4:3*h//4, w//4:3*w//4]
        edge_region = np.concatenate([
            edges[:h//4, :].flatten(),
            edges[3*h//4:, :].flatten()
        ])

        if center_region.size > 0 and edge_region.size > 0:
            center_density = np.sum(center_region > 0) / center_region.size
            edge_density = np.sum(edge_region > 0) / edge_region.size
            center_score = center_density - edge_density
        else:
            center_score = 0

        return (bg_score * 0.5 + max(0, center_score) * 0.5)

    def _find_content_bbox(self, edges) -> Optional[Tuple[int, int, int, int]]:
        """Find bounding box of content."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get overall bounding box
        x_min, y_min = edges.shape[1], edges.shape[0]
        x_max, y_max = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(edges.shape[1], x_max + padding)
        y_max = min(edges.shape[0], y_max + padding)

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _crop_region(self, frame, bbox: Tuple[int, int, int, int]) -> bytes:
        """Crop and encode region for OCR."""
        x, y, w, h = bbox
        region = frame[y:y+h, x:x+w]
        _, encoded = cv2.imencode('.jpg', region)
        return encoded.tobytes()


class CodeDetector(ContentDetector):
    """
    Detect code via dark backgrounds + monospace patterns.

    Detection strategy:
    1. Check for dark background regions (IDE-like)
    2. Detect consistent indentation patterns
    3. Look for syntax highlighting colors
    4. Check for line-based structure
    """

    # Dark background threshold
    DARK_THRESHOLD = 80

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self._cv2_available = _load_cv2()

    def detect(self, frame) -> Optional[DetectionResult]:
        """Detect code in frame."""
        if not self._cv2_available:
            return None

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Check for dark background
            dark_score = self._detect_dark_background(gray)

            # Check for line structure (code has regular lines)
            line_score = self._detect_line_structure(gray)

            # Check for syntax highlighting colors
            color_score = self._detect_syntax_colors(frame)

            # Check for indentation patterns
            indent_score = self._detect_indentation(gray)

            # Combined confidence
            confidence = (
                dark_score * 0.3 +
                line_score * 0.3 +
                color_score * 0.2 +
                indent_score * 0.2
            )

            if confidence >= self.min_confidence:
                return DetectionResult(
                    detected=True,
                    content_type="code",
                    confidence=confidence,
                    bounding_box=(0, 0, w, h),
                    region_image=self._encode_frame(frame)
                )

            return None

        except Exception as e:
            logger.debug(f"Code detection error: {e}")
            return None

    def _detect_dark_background(self, gray) -> float:
        """Detect dark background (common in IDEs)."""
        mean_brightness = np.mean(gray)

        if mean_brightness < self.DARK_THRESHOLD:
            return 1.0
        elif mean_brightness < 120:
            return 0.5
        else:
            return 0.0

    def _detect_line_structure(self, gray) -> float:
        """Detect regular horizontal line structure."""
        # Sum each row
        row_sums = np.sum(gray, axis=1)

        # Check for regularity (code has consistent line heights)
        if len(row_sums) < 10:
            return 0.0

        # Compute differences between row sums
        diffs = np.abs(np.diff(row_sums))
        variance = np.var(diffs)

        # Low variance means regular structure
        return min(1.0, 10000 / (variance + 1))

    def _detect_syntax_colors(self, frame) -> float:
        """Detect syntax highlighting colors."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Common syntax colors (blue, green, orange, purple)
        color_ranges = [
            ((100, 100, 100), (130, 255, 255)),  # Blue
            ((40, 100, 100), (80, 255, 255)),    # Green
            ((10, 100, 100), (25, 255, 255)),    # Orange
            ((130, 100, 100), (160, 255, 255)),  # Purple
        ]

        color_count = 0
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if np.sum(mask > 0) > (frame.shape[0] * frame.shape[1] * 0.01):
                color_count += 1

        return color_count / len(color_ranges)

    def _detect_indentation(self, gray) -> float:
        """Detect consistent indentation (left margin patterns)."""
        # Look at left edge of image
        left_margin = gray[:, :gray.shape[1]//4]

        # Threshold
        _, binary = cv2.threshold(left_margin, 127, 255, cv2.THRESH_BINARY)

        # Check for vertical line patterns (indentation levels)
        col_sums = np.sum(binary, axis=0)

        # Check for distinct levels
        levels = np.where(np.diff(col_sums) > np.max(col_sums) * 0.1)[0]

        if len(levels) >= 2:
            return 1.0
        elif len(levels) == 1:
            return 0.5
        return 0.0

    def _encode_frame(self, frame) -> bytes:
        """Encode frame to JPEG."""
        _, encoded = cv2.imencode('.jpg', frame)
        return encoded.tobytes()


class TextDetector(ContentDetector):
    """
    Detect general text via paragraph structure.

    Detection strategy:
    1. Look for horizontal text lines
    2. Check for paragraph structure
    3. Detect reading-like layout
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self._cv2_available = _load_cv2()

    def detect(self, frame) -> Optional[DetectionResult]:
        """Detect text in frame."""
        if not self._cv2_available:
            return None

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Detect text-like structure
            text_score = self._detect_text_structure(edges)

            # Detect paragraph blocks
            para_score = self._detect_paragraphs(edges)

            confidence = (text_score * 0.6 + para_score * 0.4)

            if confidence >= self.min_confidence:
                bbox = self._find_text_bbox(edges)

                return DetectionResult(
                    detected=True,
                    content_type="text",
                    confidence=confidence,
                    bounding_box=bbox,
                    region_image=self._crop_region(frame, bbox) if bbox else None
                )

            return None

        except Exception as e:
            logger.debug(f"Text detection error: {e}")
            return None

    def _detect_text_structure(self, edges) -> float:
        """Detect horizontal text line structure."""
        # Use morphological operations to find text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours (potential text lines)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for horizontal lines
        line_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > h * 3:  # Horizontal aspect ratio
                line_count += 1

        return min(1.0, line_count / 10)

    def _detect_paragraphs(self, edges) -> float:
        """Detect paragraph-like blocks."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for large rectangular blocks
        block_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > (edges.shape[0] * edges.shape[1] * 0.05):  # At least 5% of frame
                block_count += 1

        return min(1.0, block_count / 3)

    def _find_text_bbox(self, edges) -> Optional[Tuple[int, int, int, int]]:
        """Find bounding box of text region."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get overall bounding box
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        return (x, y, w, h)

    def _crop_region(self, frame, bbox: Tuple[int, int, int, int]) -> bytes:
        """Crop and encode region."""
        x, y, w, h = bbox
        region = frame[y:y+h, x:x+w]
        _, encoded = cv2.imencode('.jpg', region)
        return encoded.tobytes()


# Factory function
def create_detectors(
    equation: bool = True,
    code: bool = True,
    text: bool = True
) -> List[ContentDetector]:
    """Create list of enabled detectors."""
    detectors = []

    if equation:
        detectors.append(EquationDetector())
    if code:
        detectors.append(CodeDetector())
    if text:
        detectors.append(TextDetector())

    return detectors


# Test
if __name__ == "__main__":
    print("=== EDITH Detectors Test ===\n")

    if not _load_cv2():
        print("OpenCV not available. Install with: pip install opencv-python")
    else:
        print("OpenCV loaded successfully")

        # Test with a sample image if available
        detectors = create_detectors()
        print(f"Created {len(detectors)} detectors:")
        for d in detectors:
            print(f"  - {d.__class__.__name__}")
