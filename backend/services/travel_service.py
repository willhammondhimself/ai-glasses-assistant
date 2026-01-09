"""Travel Service - TripIt + FlightAware integration for travel assistance."""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Flight:
    """A flight with status information."""
    airline: str
    flight_number: str
    departure_airport: str
    arrival_airport: str
    departure_time: datetime
    arrival_time: datetime
    status: str = "On Time"         # On Time, Delayed, Cancelled, Boarding, Landed
    gate: Optional[str] = None
    terminal: Optional[str] = None
    delay_minutes: int = 0
    baggage_claim: Optional[str] = None

    def to_voice_summary(self) -> str:
        """Create a voice-friendly summary."""
        dep_time = self.departure_time.strftime("%-I:%M %p")
        arr_time = self.arrival_time.strftime("%-I:%M %p")

        parts = [f"{self.airline} {self.flight_number}"]
        parts.append(f"from {self.departure_airport} to {self.arrival_airport}")
        parts.append(f"departing {dep_time}")

        if self.status == "Delayed":
            parts.append(f"delayed {self.delay_minutes} minutes")
        elif self.status != "On Time":
            parts.append(self.status.lower())

        if self.gate:
            parts.append(f"Gate {self.gate}")
        if self.terminal:
            parts.append(f"Terminal {self.terminal}")

        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "airline": self.airline,
            "flight_number": self.flight_number,
            "departure_airport": self.departure_airport,
            "arrival_airport": self.arrival_airport,
            "departure_time": self.departure_time.isoformat(),
            "arrival_time": self.arrival_time.isoformat(),
            "status": self.status,
            "gate": self.gate,
            "terminal": self.terminal,
            "delay_minutes": self.delay_minutes,
            "baggage_claim": self.baggage_claim
        }


@dataclass
class Hotel:
    """Hotel reservation details."""
    name: str
    address: str
    check_in: datetime
    check_out: datetime
    confirmation_number: Optional[str] = None
    room_type: Optional[str] = None

    def to_voice_summary(self) -> str:
        """Create a voice-friendly summary."""
        nights = (self.check_out - self.check_in).days
        check_in_str = self.check_in.strftime("%B %d")
        return f"{self.name} for {nights} nights, checking in {check_in_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "address": self.address,
            "check_in": self.check_in.isoformat(),
            "check_out": self.check_out.isoformat(),
            "confirmation_number": self.confirmation_number,
            "room_type": self.room_type
        }


@dataclass
class Trip:
    """A complete trip itinerary."""
    id: str
    name: str
    start_date: datetime
    end_date: datetime
    flights: List[Flight] = field(default_factory=list)
    hotels: List[Hotel] = field(default_factory=list)
    car_rentals: List[Dict] = field(default_factory=list)

    def to_voice_summary(self) -> str:
        """Create a voice-friendly summary."""
        start = self.start_date.strftime("%B %d")
        end = self.end_date.strftime("%B %d")
        days = (self.end_date - self.start_date).days

        parts = [f"{self.name}"]
        parts.append(f"{start} to {end}")
        parts.append(f"{days} days")

        if self.flights:
            parts.append(f"{len(self.flights)} flights")
        if self.hotels:
            parts.append(f"{len(self.hotels)} hotel{'s' if len(self.hotels) > 1 else ''}")

        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "flights": [f.to_dict() for f in self.flights],
            "hotels": [h.to_dict() for h in self.hotels],
            "car_rentals": self.car_rentals
        }


class TravelService:
    """TripIt + FlightAware integration for travel management."""

    def __init__(self):
        self.tripit_token = os.getenv("TRIPIT_API_TOKEN")
        self.flightaware_key = os.getenv("FLIGHTAWARE_API_KEY")

        # API base URLs
        self._tripit_url = "https://api.tripit.com/v1"
        self._flightaware_url = "https://aeroapi.flightaware.com/aeroapi"

        self._client: Optional[httpx.AsyncClient] = None

        # Cache
        self._trips_cache: Optional[List[Trip]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)

    def is_configured(self) -> bool:
        """Check if travel APIs are configured."""
        return bool(self.tripit_token) or bool(self.flightaware_key)

    def tripit_configured(self) -> bool:
        """Check if TripIt is configured."""
        return bool(self.tripit_token)

    def flightaware_configured(self) -> bool:
        """Check if FlightAware is configured."""
        return bool(self.flightaware_key)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def get_upcoming_trips(self, days: int = 30) -> List[Trip]:
        """Get upcoming trips.

        Args:
            days: Number of days to look ahead

        Returns:
            List of Trip objects
        """
        # Check cache
        if self._trips_cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_ttl:
                return self._trips_cache

        if self.tripit_configured():
            trips = await self._fetch_tripit_trips(days)
            if trips:
                self._trips_cache = trips
                self._cache_time = datetime.now()
                return trips

        # Return mock data if no API configured
        return self._get_mock_trips()

    async def _fetch_tripit_trips(self, days: int) -> List[Trip]:
        """Fetch trips from TripIt API."""
        try:
            client = await self._get_client()

            # TripIt uses OAuth 1.0a - simplified for demo
            headers = {"Authorization": f"Bearer {self.tripit_token}"}

            response = await client.get(
                f"{self._tripit_url}/list/trip",
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            trips = []

            for trip_data in data.get("Trip", []):
                trip = self._parse_tripit_trip(trip_data)
                if trip:
                    trips.append(trip)

            return trips

        except Exception as e:
            logger.warning(f"TripIt API error: {e}")
            return []

    def _parse_tripit_trip(self, data: Dict) -> Optional[Trip]:
        """Parse TripIt trip data."""
        try:
            start = datetime.strptime(data["start_date"], "%Y-%m-%d")
            end = datetime.strptime(data["end_date"], "%Y-%m-%d")

            flights = []
            for air in data.get("AirObject", []):
                for segment in air.get("Segment", []):
                    flight = Flight(
                        airline=segment.get("marketing_airline", ""),
                        flight_number=segment.get("marketing_flight_number", ""),
                        departure_airport=segment.get("start_airport_code", ""),
                        arrival_airport=segment.get("end_airport_code", ""),
                        departure_time=datetime.fromisoformat(segment.get("StartDateTime", {}).get("date", "")),
                        arrival_time=datetime.fromisoformat(segment.get("EndDateTime", {}).get("date", ""))
                    )
                    flights.append(flight)

            hotels = []
            for hotel_data in data.get("LodgingObject", []):
                hotel = Hotel(
                    name=hotel_data.get("supplier_name", ""),
                    address=hotel_data.get("Address", {}).get("address", ""),
                    check_in=datetime.strptime(hotel_data.get("StartDateTime", {}).get("date", ""), "%Y-%m-%d"),
                    check_out=datetime.strptime(hotel_data.get("EndDateTime", {}).get("date", ""), "%Y-%m-%d"),
                    confirmation_number=hotel_data.get("supplier_conf_num")
                )
                hotels.append(hotel)

            return Trip(
                id=str(data.get("id", "")),
                name=data.get("display_name", "Trip"),
                start_date=start,
                end_date=end,
                flights=flights,
                hotels=hotels
            )

        except Exception as e:
            logger.warning(f"Error parsing TripIt trip: {e}")
            return None

    async def get_active_trip(self) -> Optional[Trip]:
        """Get currently active trip.

        Returns:
            Active Trip or None
        """
        trips = await self.get_upcoming_trips()
        now = datetime.now()

        for trip in trips:
            if trip.start_date <= now <= trip.end_date:
                return trip

        return None

    async def get_next_flight(self) -> Optional[Flight]:
        """Get next upcoming flight.

        Returns:
            Next Flight or None
        """
        trips = await self.get_upcoming_trips()
        now = datetime.now()

        all_flights = []
        for trip in trips:
            all_flights.extend(trip.flights)

        # Sort by departure time
        future_flights = [f for f in all_flights if f.departure_time > now]
        future_flights.sort(key=lambda f: f.departure_time)

        return future_flights[0] if future_flights else None

    async def get_flight_status(self, flight_number: str, date: Optional[datetime] = None) -> Optional[Flight]:
        """Get real-time flight status from FlightAware.

        Args:
            flight_number: Flight number (e.g., "UA1234")
            date: Optional flight date

        Returns:
            Flight with status or None
        """
        if not self.flightaware_configured():
            logger.warning("FlightAware not configured")
            return None

        try:
            client = await self._get_client()

            # Parse airline code and number
            airline_code = flight_number[:2].upper()
            number = flight_number[2:]

            headers = {"x-apikey": self.flightaware_key}

            response = await client.get(
                f"{self._flightaware_url}/flights/{airline_code}{number}",
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            flights = data.get("flights", [])

            if not flights:
                return None

            # Get most relevant flight
            flight_data = flights[0]

            return Flight(
                airline=flight_data.get("operator", {}).get("name", airline_code),
                flight_number=flight_number,
                departure_airport=flight_data.get("origin", {}).get("code", ""),
                arrival_airport=flight_data.get("destination", {}).get("code", ""),
                departure_time=datetime.fromisoformat(flight_data.get("scheduled_off", "")),
                arrival_time=datetime.fromisoformat(flight_data.get("scheduled_on", "")),
                status=self._parse_flight_status(flight_data),
                gate=flight_data.get("gate_origin"),
                terminal=flight_data.get("terminal_origin"),
                delay_minutes=flight_data.get("departure_delay", 0) // 60,
                baggage_claim=flight_data.get("baggage_claim")
            )

        except Exception as e:
            logger.warning(f"FlightAware API error: {e}")
            return None

    def _parse_flight_status(self, data: Dict) -> str:
        """Parse flight status from FlightAware data."""
        status = data.get("status", "").lower()

        if "cancelled" in status:
            return "Cancelled"
        if "landed" in status:
            return "Landed"
        if "boarding" in status:
            return "Boarding"
        if "delayed" in status or data.get("departure_delay", 0) > 900:  # >15 min
            return "Delayed"

        return "On Time"

    async def check_flight_delays(self) -> List[Flight]:
        """Check for delayed upcoming flights.

        Returns:
            List of delayed flights
        """
        trips = await self.get_upcoming_trips()
        delayed = []

        for trip in trips:
            for flight in trip.flights:
                if flight.departure_time > datetime.now():
                    # Get real-time status
                    status = await self.get_flight_status(
                        f"{flight.airline[:2]}{flight.flight_number}"
                    )
                    if status and status.status == "Delayed":
                        delayed.append(status)

        return delayed

    async def get_hotel_info(self, trip_id: str) -> Optional[Hotel]:
        """Get hotel info for a trip.

        Args:
            trip_id: Trip ID

        Returns:
            Hotel or None
        """
        trips = await self.get_upcoming_trips()

        for trip in trips:
            if trip.id == trip_id and trip.hotels:
                return trip.hotels[0]

        return None

    def _get_mock_trips(self) -> List[Trip]:
        """Get mock trips when APIs not configured."""
        now = datetime.now()

        mock_flight = Flight(
            airline="United",
            flight_number="UA1234",
            departure_airport="SFO",
            arrival_airport="LAS",
            departure_time=now + timedelta(days=3, hours=14, minutes=30),
            arrival_time=now + timedelta(days=3, hours=16, minutes=45),
            status="On Time",
            gate="B42",
            terminal="B"
        )

        mock_hotel = Hotel(
            name="Bellagio Las Vegas",
            address="3600 S Las Vegas Blvd, Las Vegas, NV",
            check_in=now + timedelta(days=3),
            check_out=now + timedelta(days=6),
            confirmation_number="BEL123456"
        )

        mock_trip = Trip(
            id="mock-1",
            name="Vegas Poker Trip",
            start_date=now + timedelta(days=3),
            end_date=now + timedelta(days=6),
            flights=[mock_flight],
            hotels=[mock_hotel]
        )

        return [mock_trip]

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global instance
_travel_service: Optional[TravelService] = None


def get_travel_service() -> TravelService:
    """Get or create global travel service."""
    global _travel_service
    if _travel_service is None:
        _travel_service = TravelService()
    return _travel_service
