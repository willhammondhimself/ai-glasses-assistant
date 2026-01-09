"""Travel Assistant voice tool - flight status, itineraries, trip info."""
import logging
import re
from datetime import datetime, timedelta
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class TravelVoiceTool(VoiceTool):
    """Voice-controlled travel assistant via TripIt + FlightAware."""

    name = "travel"
    description = "Travel info - flights, hotels, itineraries, trip status"

    keywords = [
        r"\bflight\b",
        r"\btrip\b",
        r"\btravel\b",
        r"\bairport\b",
        r"\bgate\b",
        r"\bboarding\b",
        r"\bhotel\b",
        r"\bitinerary\b",
        r"\bwhen\s+(do\s+)?i\s+(leave|fly|depart)\b",
        r"\b(flight|plane)\s+(status|time|delayed?)\b",
        r"\bcheck\s+in\b",
        r"\bbaggage\b",
        r"\bterminal\b",
    ]

    priority = 8

    def __init__(self):
        self._travel_service = None

    def _get_service(self):
        """Get travel service (lazy load)."""
        if self._travel_service is None:
            from backend.services.travel_service import get_travel_service
            self._travel_service = get_travel_service()
        return self._travel_service

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute travel command.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with travel info
        """
        query_lower = query.lower()
        service = self._get_service()

        try:
            # Flight status check
            if self._is_flight_status_query(query_lower):
                flight_num = self._extract_flight_number(query)
                return await self._handle_flight_status(service, flight_num)

            # Next flight query
            if self._is_next_flight_query(query_lower):
                return await self._handle_next_flight(service)

            # Gate/terminal query
            if self._is_gate_query(query_lower):
                return await self._handle_gate_info(service)

            # Hotel query
            if self._is_hotel_query(query_lower):
                return await self._handle_hotel(service)

            # Itinerary query
            if self._is_itinerary_query(query_lower):
                return await self._handle_itinerary(service)

            # Delay check
            if self._is_delay_query(query_lower):
                return await self._handle_delays(service)

            # Upcoming trips
            if self._is_trips_query(query_lower):
                return await self._handle_trips(service)

            # Default: show next flight or upcoming trip
            return await self._handle_summary(service)

        except Exception as e:
            logger.error(f"Travel tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble getting your travel info.",
                data={"error": str(e)}
            )

    def _is_flight_status_query(self, query: str) -> bool:
        patterns = ["flight status", "is my flight", "flight delayed", "on time"]
        return any(p in query for p in patterns)

    def _is_next_flight_query(self, query: str) -> bool:
        patterns = ["my flight", "when do i fly", "when do i leave", "when's my flight", "next flight"]
        return any(p in query for p in patterns)

    def _is_gate_query(self, query: str) -> bool:
        patterns = ["gate", "terminal", "where do i go", "which gate"]
        return any(p in query for p in patterns)

    def _is_hotel_query(self, query: str) -> bool:
        patterns = ["hotel", "where am i staying", "check in", "reservation"]
        return any(p in query for p in patterns)

    def _is_itinerary_query(self, query: str) -> bool:
        patterns = ["itinerary", "schedule", "trip details", "travel plan"]
        return any(p in query for p in patterns)

    def _is_delay_query(self, query: str) -> bool:
        patterns = ["delay", "delayed", "any delays"]
        return any(p in query for p in patterns)

    def _is_trips_query(self, query: str) -> bool:
        patterns = ["upcoming trip", "my trips", "travel plans", "where am i going"]
        return any(p in query for p in patterns)

    def _extract_flight_number(self, query: str) -> str:
        """Extract flight number from query."""
        # Match patterns like "UA1234", "United 1234", "AA 456"
        match = re.search(r"([A-Z]{2})\s*(\d{1,4})", query.upper())
        if match:
            return f"{match.group(1)}{match.group(2)}"

        # Try airline name + number
        airlines = {
            "united": "UA", "delta": "DL", "american": "AA",
            "southwest": "WN", "jetblue": "B6", "alaska": "AS"
        }
        for name, code in airlines.items():
            if name in query.lower():
                num_match = re.search(r"(\d{1,4})", query)
                if num_match:
                    return f"{code}{num_match.group(1)}"

        return ""

    async def _handle_flight_status(self, service, flight_num: str) -> VoiceToolResult:
        """Handle flight status query."""
        if flight_num:
            flight = await service.get_flight_status(flight_num)
        else:
            # Get next flight
            flight = await service.get_next_flight()

        if not flight:
            return VoiceToolResult(
                success=False,
                message="I couldn't find that flight. Try specifying the flight number like 'UA 1234'.",
                data={"error": "flight_not_found"}
            )

        # Build status message
        if flight.status == "On Time":
            message = f"{flight.airline} {flight.flight_number} is on time. "
        elif flight.status == "Delayed":
            message = f"{flight.airline} {flight.flight_number} is delayed {flight.delay_minutes} minutes. "
        else:
            message = f"{flight.airline} {flight.flight_number} is {flight.status.lower()}. "

        dep_time = flight.departure_time.strftime("%-I:%M %p")
        message += f"Departing at {dep_time}"

        if flight.gate:
            message += f" from Gate {flight.gate}"
        if flight.terminal:
            message += f", Terminal {flight.terminal}"

        message += "."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"flight": flight.to_dict()}
        )

    async def _handle_next_flight(self, service) -> VoiceToolResult:
        """Handle next flight query."""
        flight = await service.get_next_flight()

        if not flight:
            return VoiceToolResult(
                success=True,
                message="You don't have any upcoming flights.",
                data={"has_flights": False}
            )

        # Calculate time until departure
        now = datetime.now()
        time_until = flight.departure_time - now

        if time_until.days > 0:
            time_str = f"in {time_until.days} days"
        elif time_until.seconds > 3600:
            hours = time_until.seconds // 3600
            time_str = f"in {hours} hours"
        else:
            minutes = time_until.seconds // 60
            time_str = f"in {minutes} minutes"

        dep_time = flight.departure_time.strftime("%-I:%M %p on %B %d")
        message = (
            f"Your next flight is {flight.airline} {flight.flight_number} "
            f"to {flight.arrival_airport}, departing {time_str} at {dep_time}."
        )

        if flight.gate:
            message += f" Gate {flight.gate}."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"flight": flight.to_dict(), "time_until": str(time_until)}
        )

    async def _handle_gate_info(self, service) -> VoiceToolResult:
        """Handle gate/terminal query."""
        flight = await service.get_next_flight()

        if not flight:
            return VoiceToolResult(
                success=False,
                message="No upcoming flights found.",
                data={"error": "no_flights"}
            )

        if flight.gate:
            message = f"Your flight departs from Gate {flight.gate}"
            if flight.terminal:
                message += f", Terminal {flight.terminal}"
            message += "."
        else:
            message = "Gate information isn't available yet. Check back closer to departure."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"flight": flight.to_dict()}
        )

    async def _handle_hotel(self, service) -> VoiceToolResult:
        """Handle hotel query."""
        trip = await service.get_active_trip()

        if not trip:
            trips = await service.get_upcoming_trips()
            if trips:
                trip = trips[0]

        if not trip or not trip.hotels:
            return VoiceToolResult(
                success=True,
                message="No hotel reservations found.",
                data={"has_hotel": False}
            )

        hotel = trip.hotels[0]
        check_in = hotel.check_in.strftime("%B %d")
        check_out = hotel.check_out.strftime("%B %d")
        nights = (hotel.check_out - hotel.check_in).days

        message = f"You're staying at {hotel.name} for {nights} nights, {check_in} to {check_out}."

        if hotel.confirmation_number:
            message += f" Confirmation: {hotel.confirmation_number}."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"hotel": hotel.to_dict()}
        )

    async def _handle_itinerary(self, service) -> VoiceToolResult:
        """Handle full itinerary query."""
        trip = await service.get_active_trip()

        if not trip:
            trips = await service.get_upcoming_trips()
            if trips:
                trip = trips[0]

        if not trip:
            return VoiceToolResult(
                success=True,
                message="No trips found. Add trips to TripIt to see them here.",
                data={"has_trips": False}
            )

        message = f"Your {trip.name}: "

        # Add flights
        if trip.flights:
            outbound = trip.flights[0]
            message += f"Fly to {outbound.arrival_airport} on {outbound.departure_time.strftime('%B %d')}. "

            if len(trip.flights) > 1:
                return_flight = trip.flights[-1]
                message += f"Return on {return_flight.departure_time.strftime('%B %d')}. "

        # Add hotel
        if trip.hotels:
            hotel = trip.hotels[0]
            nights = (hotel.check_out - hotel.check_in).days
            message += f"{nights} nights at {hotel.name}."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"trip": trip.to_dict()}
        )

    async def _handle_delays(self, service) -> VoiceToolResult:
        """Handle delay check query."""
        delayed = await service.check_flight_delays()

        if not delayed:
            return VoiceToolResult(
                success=True,
                message="Good news! All your upcoming flights are on time.",
                data={"delays": []}
            )

        if len(delayed) == 1:
            flight = delayed[0]
            message = f"Your flight {flight.airline} {flight.flight_number} is delayed {flight.delay_minutes} minutes."
        else:
            message = f"You have {len(delayed)} delayed flights. "
            message += f"First: {delayed[0].airline} {delayed[0].flight_number}, delayed {delayed[0].delay_minutes} minutes."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"delays": [f.to_dict() for f in delayed]}
        )

    async def _handle_trips(self, service) -> VoiceToolResult:
        """Handle upcoming trips query."""
        trips = await service.get_upcoming_trips()

        if not trips:
            return VoiceToolResult(
                success=True,
                message="You don't have any upcoming trips.",
                data={"trips": []}
            )

        if len(trips) == 1:
            trip = trips[0]
            start = trip.start_date.strftime("%B %d")
            message = f"You have one upcoming trip: {trip.name} starting {start}."
        else:
            message = f"You have {len(trips)} upcoming trips. "
            message += f"Next: {trips[0].name} on {trips[0].start_date.strftime('%B %d')}."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"trips": [t.to_dict() for t in trips]}
        )

    async def _handle_summary(self, service) -> VoiceToolResult:
        """Handle general travel summary."""
        # Check for active trip
        active = await service.get_active_trip()
        if active:
            message = f"You're on your {active.name} trip. "
            if active.flights:
                next_flight = [f for f in active.flights if f.departure_time > datetime.now()]
                if next_flight:
                    message += f"Next flight: {next_flight[0].to_voice_summary()}"
                else:
                    message += "All flights completed."
            return VoiceToolResult(
                success=True,
                message=message,
                data={"active_trip": active.to_dict()}
            )

        # Check for upcoming flights
        next_flight = await service.get_next_flight()
        if next_flight:
            days_until = (next_flight.departure_time - datetime.now()).days
            if days_until == 0:
                message = f"You fly today! {next_flight.to_voice_summary()}"
            elif days_until == 1:
                message = f"You fly tomorrow: {next_flight.to_voice_summary()}"
            else:
                message = f"Next flight in {days_until} days: {next_flight.to_voice_summary()}"

            return VoiceToolResult(
                success=True,
                message=message,
                data={"next_flight": next_flight.to_dict()}
            )

        return VoiceToolResult(
            success=True,
            message="No upcoming travel plans.",
            data={"has_travel": False}
        )
