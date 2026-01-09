"""Package tracking voice tool."""
import re
import logging
from typing import Optional
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class PackageVoiceTool(VoiceTool):
    """Voice-controlled package tracking."""

    name = "packages"
    description = "Track package deliveries and shipments"

    keywords = [
        r"\bpackage[s]?\b",
        r"\bdelivery\b",
        r"\bdeliveries\b",
        r"\bshipping\b",
        r"\bshipment[s]?\b",
        r"\btrack(?:ing)?\b",
        r"\border(?:s)?\b",
        r"\bamazon\b",
        r"\bfedex\b",
        r"\bups\b",
        r"\busps\b",
        r"\bwhere('s| is)\s+my\b",
        r"\bwhen\s+will.*arrive\b",
    ]

    priority = 6

    def __init__(self):
        self._tracker = None
        self._simple_tracker = None

    def _get_tracker(self):
        """Get package tracker (lazy load)."""
        if self._tracker is None:
            from backend.services.package_tracker import get_package_tracker
            self._tracker = get_package_tracker()
        return self._tracker

    def _get_simple_tracker(self):
        """Get simple tracker for manual tracking."""
        if self._simple_tracker is None:
            from backend.services.package_tracker import get_simple_tracker
            self._simple_tracker = get_simple_tracker()
        return self._simple_tracker

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute package tracking command.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with package info
        """
        query_lower = query.lower()
        tracker = self._get_tracker()

        try:
            # Check for tracking number in query
            tracking_number = self._extract_tracking_number(query)
            if tracking_number:
                return await self._handle_track_specific(tracking_number, query)

            # Check for carrier-specific query
            carrier = self._extract_carrier(query_lower)
            if carrier:
                return await self._handle_carrier_query(carrier)

            # "Any packages today?" / "Arriving today?"
            if self._is_today_query(query_lower):
                return await self._handle_arriving_today()

            # "Where's my package?" / "Any packages?"
            if self._is_status_query(query_lower):
                return await self._handle_status_all()

            # Default: show all active packages
            return await self._handle_status_all()

        except Exception as e:
            logger.error(f"Package tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble checking your packages.",
                data={"error": str(e)}
            )

    def _is_today_query(self, query: str) -> bool:
        patterns = ["today", "arriving today", "expected today", "coming today"]
        return any(p in query for p in patterns)

    def _is_status_query(self, query: str) -> bool:
        patterns = [
            "where", "any package", "my package", "my order",
            "status", "check package", "check my"
        ]
        return any(p in query for p in patterns)

    def _extract_tracking_number(self, query: str) -> Optional[str]:
        """Extract tracking number from query."""
        # Common tracking number patterns
        patterns = [
            r'\b(1Z[A-Z0-9]{16})\b',  # UPS
            r'\b(\d{12,22})\b',  # FedEx, USPS
            r'\b([A-Z]{2}\d{9}[A-Z]{2})\b',  # International
            r'track(?:ing)?\s+(?:number\s+)?([A-Z0-9]{10,30})',  # General
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def _extract_carrier(self, query: str) -> Optional[str]:
        """Extract carrier name from query."""
        carriers = {
            "ups": "ups",
            "fedex": "fedex",
            "fed ex": "fedex",
            "usps": "usps",
            "postal": "usps",
            "amazon": "amazon",
            "dhl": "dhl",
        }

        for name, slug in carriers.items():
            if name in query:
                return slug

        return None

    async def _handle_arriving_today(self) -> VoiceToolResult:
        """Handle query about packages arriving today."""
        tracker = self._get_tracker()

        if not tracker.is_configured():
            return await self._handle_simple_status()

        packages = await tracker.get_packages_arriving_today()

        if not packages:
            return VoiceToolResult(
                success=True,
                message="No packages expected today.",
                data={"arriving_today": 0}
            )

        if len(packages) == 1:
            pkg = packages[0]
            return VoiceToolResult(
                success=True,
                message=pkg.to_voice_summary(),
                data={"arriving_today": 1, "package": pkg.to_dict()}
            )

        # Multiple packages
        summaries = [pkg.title or pkg.carrier for pkg in packages]
        message = f"You have {len(packages)} packages arriving today: " + ", ".join(summaries)

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "arriving_today": len(packages),
                "packages": [pkg.to_dict() for pkg in packages]
            }
        )

    async def _handle_status_all(self) -> VoiceToolResult:
        """Handle general package status query."""
        tracker = self._get_tracker()

        if not tracker.is_configured():
            return await self._handle_simple_status()

        packages = await tracker.get_packages_in_transit()

        if not packages:
            return VoiceToolResult(
                success=True,
                message="You have no packages in transit.",
                data={"in_transit": 0}
            )

        if len(packages) == 1:
            pkg = packages[0]
            return VoiceToolResult(
                success=True,
                message=pkg.to_voice_summary(),
                data={"in_transit": 1, "package": pkg.to_dict()}
            )

        # Multiple packages - summarize
        out_for_delivery = [p for p in packages if p.status.value == "out_for_delivery"]

        if out_for_delivery:
            message = f"You have {len(packages)} packages in transit. {len(out_for_delivery)} out for delivery!"
        else:
            message = f"You have {len(packages)} packages in transit."

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "in_transit": len(packages),
                "out_for_delivery": len(out_for_delivery),
                "packages": [pkg.to_dict() for pkg in packages]
            }
        )

    async def _handle_simple_status(self) -> VoiceToolResult:
        """Handle status using simple tracker (no API)."""
        simple = self._get_simple_tracker()
        packages = await simple.get_packages()

        if not packages:
            return VoiceToolResult(
                success=True,
                message="No packages being tracked. Tell me a tracking number to add one.",
                data={"tracked": 0}
            )

        if len(packages) == 1:
            pkg = packages[0]
            return VoiceToolResult(
                success=True,
                message=f"Tracking one package from {pkg.carrier}. For detailed status, add an AfterShip API key.",
                data={"tracked": 1, "package": pkg.to_dict()}
            )

        return VoiceToolResult(
            success=True,
            message=f"Tracking {len(packages)} packages. For detailed status, add an AfterShip API key.",
            data={"tracked": len(packages), "packages": [p.to_dict() for p in packages]}
        )

    async def _handle_track_specific(self, tracking_number: str, query: str) -> VoiceToolResult:
        """Handle tracking a specific package."""
        tracker = self._get_tracker()
        carrier = self._extract_carrier(query.lower())

        if tracker.is_configured():
            # Try to get existing tracking
            package = await tracker.get_package(tracking_number, carrier)

            if not package:
                # Try to add it
                package = await tracker.track_package(tracking_number, carrier)

            if package:
                return VoiceToolResult(
                    success=True,
                    message=package.to_voice_summary(),
                    data={"package": package.to_dict()}
                )

            return VoiceToolResult(
                success=False,
                message=f"Couldn't find tracking info for {tracking_number}.",
                data={"tracking_number": tracking_number}
            )

        # Use simple tracker
        simple = self._get_simple_tracker()
        carrier = carrier or "unknown carrier"
        package = await simple.add_package(tracking_number, carrier)

        return VoiceToolResult(
            success=True,
            message=f"Added package {tracking_number} from {carrier} to tracking. For live updates, configure AfterShip API.",
            data={"package": package.to_dict(), "needs_api": True}
        )

    async def _handle_carrier_query(self, carrier: str) -> VoiceToolResult:
        """Handle query about specific carrier packages."""
        tracker = self._get_tracker()

        if not tracker.is_configured():
            return VoiceToolResult(
                success=True,
                message=f"To track {carrier} packages, add an AfterShip API key or tell me the tracking number.",
                data={"carrier": carrier, "needs_api": True}
            )

        packages = await tracker.get_active_packages()
        carrier_packages = [p for p in packages if carrier in p.carrier.lower()]

        if not carrier_packages:
            return VoiceToolResult(
                success=True,
                message=f"No {carrier.upper()} packages being tracked.",
                data={"carrier": carrier, "count": 0}
            )

        if len(carrier_packages) == 1:
            pkg = carrier_packages[0]
            return VoiceToolResult(
                success=True,
                message=f"Your {carrier.upper()} package: {pkg.to_voice_summary()}",
                data={"carrier": carrier, "package": pkg.to_dict()}
            )

        # Multiple packages
        summaries = [pkg.to_voice_summary() for pkg in carrier_packages[:3]]
        message = f"You have {len(carrier_packages)} {carrier.upper()} packages. " + " ".join(summaries)

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "carrier": carrier,
                "count": len(carrier_packages),
                "packages": [p.to_dict() for p in carrier_packages]
            }
        )
