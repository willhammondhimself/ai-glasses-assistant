"""Package tracking service using AfterShip API."""
import os
import logging
import httpx
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

AFTERSHIP_API_URL = "https://api.aftership.com/v4"


class DeliveryStatus(Enum):
    """Package delivery status."""
    PENDING = "pending"
    INFO_RECEIVED = "info_received"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXCEPTION = "exception"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass
class PackageUpdate:
    """A tracking update/event."""
    message: str
    location: str
    timestamp: datetime


@dataclass
class Package:
    """Represents a tracked package."""
    id: str
    tracking_number: str
    carrier: str
    title: Optional[str]
    status: DeliveryStatus
    expected_delivery: Optional[datetime]
    last_update: Optional[str]
    last_location: Optional[str]
    origin: Optional[str]
    destination: Optional[str]
    updates: List[PackageUpdate]

    def to_voice_summary(self) -> str:
        """Return voice-friendly summary."""
        status_text = {
            DeliveryStatus.PENDING: "is pending pickup",
            DeliveryStatus.INFO_RECEIVED: "was received by the carrier",
            DeliveryStatus.IN_TRANSIT: "is in transit",
            DeliveryStatus.OUT_FOR_DELIVERY: "is out for delivery",
            DeliveryStatus.DELIVERED: "has been delivered",
            DeliveryStatus.FAILED: "delivery failed",
            DeliveryStatus.EXCEPTION: "has an exception",
            DeliveryStatus.EXPIRED: "tracking has expired",
            DeliveryStatus.UNKNOWN: "status is unknown",
        }.get(self.status, "status unknown")

        name = self.title or f"package from {self.carrier}"

        if self.expected_delivery and self.status not in [DeliveryStatus.DELIVERED, DeliveryStatus.FAILED]:
            return f"Your {name} {status_text}. Expected delivery: {self.expected_delivery.strftime('%B %d')}."
        elif self.last_location:
            return f"Your {name} {status_text}. Last seen in {self.last_location}."
        else:
            return f"Your {name} {status_text}."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tracking_number": self.tracking_number,
            "carrier": self.carrier,
            "title": self.title,
            "status": self.status.value,
            "expected_delivery": self.expected_delivery.isoformat() if self.expected_delivery else None,
            "last_update": self.last_update,
            "last_location": self.last_location,
            "origin": self.origin,
            "destination": self.destination,
            "updates": [
                {"message": u.message, "location": u.location, "timestamp": u.timestamp.isoformat()}
                for u in self.updates[:5]  # Limit updates
            ]
        }


class PackageTrackerService:
    """Package tracking service using AfterShip API."""

    # Common carrier slugs
    CARRIERS = {
        "ups": "ups",
        "fedex": "fedex",
        "usps": "usps",
        "dhl": "dhl",
        "amazon": "amazon-logistics",
        "ontrac": "ontrac",
        "lasership": "lasership",
    }

    def __init__(self):
        """Initialize package tracker."""
        self.api_key = os.environ.get("AFTERSHIP_API_KEY")
        self._packages_cache: Dict[str, Package] = {}

    def is_configured(self) -> bool:
        """Check if AfterShip API is configured."""
        return bool(self.api_key)

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make an API request to AfterShip.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body

        Returns:
            Response data or None
        """
        if not self.api_key:
            raise RuntimeError("AfterShip API key not configured")

        url = f"{AFTERSHIP_API_URL}{endpoint}"
        headers = {
            "aftership-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=data,
                timeout=15.0
            )

            if response.status_code >= 400:
                logger.error(f"AfterShip API error: {response.status_code} - {response.text}")
                return None

            return response.json()

    async def track_package(
        self,
        tracking_number: str,
        carrier: Optional[str] = None,
        title: Optional[str] = None
    ) -> Optional[Package]:
        """Add a package to track.

        Args:
            tracking_number: The tracking number
            carrier: Carrier slug (auto-detected if not provided)
            title: Friendly name for the package

        Returns:
            Package object or None
        """
        data = {
            "tracking": {
                "tracking_number": tracking_number.replace(" ", "").upper()
            }
        }

        if carrier:
            carrier_slug = self.CARRIERS.get(carrier.lower(), carrier.lower())
            data["tracking"]["slug"] = carrier_slug

        if title:
            data["tracking"]["title"] = title

        result = await self._api_request("POST", "/trackings", data)

        if not result or "data" not in result:
            return None

        return self._parse_tracking(result["data"]["tracking"])

    async def get_package(self, tracking_number: str, carrier: Optional[str] = None) -> Optional[Package]:
        """Get tracking info for a package.

        Args:
            tracking_number: The tracking number
            carrier: Carrier slug

        Returns:
            Package object or None
        """
        tracking_number = tracking_number.replace(" ", "").upper()

        if carrier:
            carrier_slug = self.CARRIERS.get(carrier.lower(), carrier.lower())
            endpoint = f"/trackings/{carrier_slug}/{tracking_number}"
        else:
            # Try to find in active trackings
            active = await self.get_active_packages()
            for pkg in active:
                if pkg.tracking_number == tracking_number:
                    return pkg
            return None

        result = await self._api_request("GET", endpoint)

        if not result or "data" not in result:
            return None

        return self._parse_tracking(result["data"]["tracking"])

    async def get_active_packages(self) -> List[Package]:
        """Get all actively tracked packages.

        Returns:
            List of Package objects
        """
        result = await self._api_request("GET", "/trackings")

        if not result or "data" not in result:
            return []

        packages = []
        for tracking in result["data"]["trackings"]:
            pkg = self._parse_tracking(tracking)
            if pkg:
                packages.append(pkg)

        return packages

    async def get_packages_arriving_today(self) -> List[Package]:
        """Get packages expected to arrive today.

        Returns:
            List of packages with today's expected delivery
        """
        packages = await self.get_active_packages()
        today = datetime.now().date()

        arriving = []
        for pkg in packages:
            if pkg.status == DeliveryStatus.OUT_FOR_DELIVERY:
                arriving.append(pkg)
            elif pkg.expected_delivery and pkg.expected_delivery.date() == today:
                arriving.append(pkg)

        return arriving

    async def get_packages_in_transit(self) -> List[Package]:
        """Get packages currently in transit.

        Returns:
            List of in-transit packages
        """
        packages = await self.get_active_packages()
        return [
            pkg for pkg in packages
            if pkg.status in [DeliveryStatus.IN_TRANSIT, DeliveryStatus.OUT_FOR_DELIVERY]
        ]

    def _parse_tracking(self, data: Dict) -> Optional[Package]:
        """Parse AfterShip tracking data into Package object."""
        try:
            # Parse status
            tag = data.get("tag", "Unknown").lower()
            status = DeliveryStatus.UNKNOWN
            for s in DeliveryStatus:
                if s.value == tag:
                    status = s
                    break

            # Parse expected delivery
            expected = None
            if data.get("expected_delivery"):
                try:
                    expected = datetime.fromisoformat(data["expected_delivery"].replace("Z", "+00:00"))
                except Exception:
                    pass

            # Parse checkpoints (updates)
            updates = []
            for cp in data.get("checkpoints", [])[:10]:
                try:
                    ts = datetime.fromisoformat(cp["checkpoint_time"].replace("Z", "+00:00"))
                    updates.append(PackageUpdate(
                        message=cp.get("message", ""),
                        location=cp.get("location", cp.get("city", "")),
                        timestamp=ts
                    ))
                except Exception:
                    continue

            # Get last update info
            last_update = None
            last_location = None
            if updates:
                last_update = updates[0].message
                last_location = updates[0].location

            return Package(
                id=data.get("id", ""),
                tracking_number=data.get("tracking_number", ""),
                carrier=data.get("slug", ""),
                title=data.get("title"),
                status=status,
                expected_delivery=expected,
                last_update=last_update,
                last_location=last_location,
                origin=data.get("origin_country_iso3"),
                destination=data.get("destination_country_iso3"),
                updates=updates
            )
        except Exception as e:
            logger.error(f"Failed to parse tracking data: {e}")
            return None

    async def detect_carrier(self, tracking_number: str) -> List[str]:
        """Detect possible carriers for a tracking number.

        Args:
            tracking_number: The tracking number

        Returns:
            List of possible carrier slugs
        """
        result = await self._api_request(
            "POST",
            "/couriers/detect",
            {"tracking": {"tracking_number": tracking_number}}
        )

        if not result or "data" not in result:
            return []

        return [c["slug"] for c in result["data"]["couriers"]]


# Simple in-memory tracking for when AfterShip isn't configured
class SimplePackageTracker:
    """Simple in-memory package tracker (no API required)."""

    def __init__(self):
        self.packages: Dict[str, Dict] = {}

    async def add_package(self, tracking_number: str, carrier: str, title: Optional[str] = None) -> Package:
        """Add a package to track manually."""
        pkg_id = tracking_number.replace(" ", "").upper()
        self.packages[pkg_id] = {
            "tracking_number": pkg_id,
            "carrier": carrier,
            "title": title or f"Package from {carrier}",
            "status": "in_transit",
            "added_at": datetime.now().isoformat()
        }

        return Package(
            id=pkg_id,
            tracking_number=pkg_id,
            carrier=carrier,
            title=title,
            status=DeliveryStatus.IN_TRANSIT,
            expected_delivery=None,
            last_update="Added to tracking",
            last_location=None,
            origin=None,
            destination=None,
            updates=[]
        )

    async def get_packages(self) -> List[Package]:
        """Get all tracked packages."""
        packages = []
        for data in self.packages.values():
            packages.append(Package(
                id=data["tracking_number"],
                tracking_number=data["tracking_number"],
                carrier=data["carrier"],
                title=data.get("title"),
                status=DeliveryStatus.IN_TRANSIT,
                expected_delivery=None,
                last_update=None,
                last_location=None,
                origin=None,
                destination=None,
                updates=[]
            ))
        return packages

    async def remove_package(self, tracking_number: str) -> bool:
        """Remove a package from tracking."""
        pkg_id = tracking_number.replace(" ", "").upper()
        if pkg_id in self.packages:
            del self.packages[pkg_id]
            return True
        return False


# Global instances
_package_tracker: Optional[PackageTrackerService] = None
_simple_tracker: Optional[SimplePackageTracker] = None


def get_package_tracker() -> PackageTrackerService:
    """Get or create global package tracker."""
    global _package_tracker
    if _package_tracker is None:
        _package_tracker = PackageTrackerService()
    return _package_tracker


def get_simple_tracker() -> SimplePackageTracker:
    """Get or create simple package tracker."""
    global _simple_tracker
    if _simple_tracker is None:
        _simple_tracker = SimplePackageTracker()
    return _simple_tracker
