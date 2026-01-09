"""Wolfram Alpha API client for advanced physics and circuit solving."""

import os
import re
import httpx
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

WOLFRAM_API_URL = "https://api.wolframalpha.com/v2/query"
WOLFRAM_SHORT_API = "https://api.wolframalpha.com/v1/result"


@dataclass
class WolframResult:
    """Result from Wolfram Alpha query."""
    success: bool
    query: str
    result: str  # Primary result text
    steps: List[str]  # Solution steps if available
    plots: List[str]  # Plot image URLs if available
    raw_pods: Dict[str, Any]  # All pod data
    error: Optional[str] = None


class WolframClient:
    """Wolfram Alpha API client for physics, circuits, and math.

    Provides structured access to Wolfram Alpha for:
    - Circuit analysis (Ohm's law, Kirchhoff, RC/RL circuits)
    - Vector operations (cross/dot products, projections)
    - Advanced calculus (PDEs, series, multi-variable)
    - Physics problems (kinematics, E&M, thermodynamics)
    """

    def __init__(self):
        self.api_key = os.getenv("WOLFRAM_APP_ID")
        if not self.api_key:
            logger.warning("WOLFRAM_APP_ID not set - Wolfram client disabled, using Gemini fallback")
        self.available = bool(self.api_key)

    async def solve(self, query: str, include_steps: bool = True) -> WolframResult:
        """Send query to Wolfram Alpha full results API.

        Args:
            query: Natural language or mathematical query
            include_steps: Whether to request step-by-step solution

        Returns:
            WolframResult with parsed response
        """
        if not self.available:
            return WolframResult(
                success=False,
                query=query,
                result="",
                steps=[],
                plots=[],
                raw_pods={},
                error="Wolfram API key not configured"
            )

        params = {
            "input": query,
            "appid": self.api_key,
            "output": "json",
            "format": "plaintext,image",
            "podstate": "Step-by-step solution" if include_steps else "",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(WOLFRAM_API_URL, params=params)
                response.raise_for_status()
                data = response.json()

            return self._parse_response(query, data)

        except httpx.HTTPStatusError as e:
            logger.error(f"Wolfram API error: {e}")
            return WolframResult(
                success=False,
                query=query,
                result="",
                steps=[],
                plots=[],
                raw_pods={},
                error=f"API error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Wolfram query failed: {e}")
            return WolframResult(
                success=False,
                query=query,
                result="",
                steps=[],
                plots=[],
                raw_pods={},
                error=str(e)
            )

    async def solve_short(self, query: str) -> str:
        """Get a short answer from Wolfram Alpha.

        Args:
            query: Natural language query

        Returns:
            Short text answer or error message
        """
        if not self.available:
            return "Wolfram API not configured"

        params = {
            "i": query,
            "appid": self.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(WOLFRAM_SHORT_API, params=params)
                if response.status_code == 501:
                    return "Wolfram couldn't understand the query"
                response.raise_for_status()
                return response.text

        except Exception as e:
            logger.error(f"Wolfram short query failed: {e}")
            return f"Error: {str(e)}"

    async def solve_circuit(self, description: str) -> WolframResult:
        """Solve a circuit problem.

        Args:
            description: Circuit description like "5V battery, 10 ohm resistor"

        Returns:
            WolframResult with circuit analysis
        """
        # Parse circuit description into Wolfram query
        query = self._build_circuit_query(description)
        return await self.solve(query)

    async def solve_vector(self, operation: str, v1: List[float], v2: Optional[List[float]] = None) -> WolframResult:
        """Solve a vector operation.

        Args:
            operation: 'cross', 'dot', 'magnitude', 'normalize', 'angle'
            v1: First vector
            v2: Second vector (for binary operations)

        Returns:
            WolframResult with vector calculation
        """
        query = self._build_vector_query(operation, v1, v2)
        return await self.solve(query)

    async def solve_ohms_law(self, **kwargs) -> WolframResult:
        """Solve Ohm's law (V = IR) given 2 of 3 values.

        Args:
            voltage: Voltage in volts (V)
            current: Current in amps (I)
            resistance: Resistance in ohms (R)

        Returns:
            WolframResult with calculated value
        """
        v = kwargs.get("voltage")
        i = kwargs.get("current")
        r = kwargs.get("resistance")

        if v is not None and i is not None:
            query = f"resistance from {v} volts and {i} amps using Ohm's law"
        elif v is not None and r is not None:
            query = f"current from {v} volts and {r} ohms using Ohm's law"
        elif i is not None and r is not None:
            query = f"voltage from {i} amps and {r} ohms using Ohm's law"
        else:
            return WolframResult(
                success=False,
                query="",
                result="",
                steps=[],
                plots=[],
                raw_pods={},
                error="Need at least 2 of: voltage, current, resistance"
            )

        return await self.solve(query)

    async def solve_series_parallel(
        self, resistors: List[float], config: str = "series"
    ) -> WolframResult:
        """Calculate equivalent resistance for series or parallel resistors.

        Args:
            resistors: List of resistance values in ohms
            config: 'series' or 'parallel'

        Returns:
            WolframResult with equivalent resistance
        """
        r_str = ", ".join(f"{r} ohms" for r in resistors)
        if config == "series":
            query = f"equivalent resistance of {r_str} in series"
        else:
            query = f"equivalent resistance of {r_str} in parallel"
        return await self.solve(query)

    async def solve_kirchhoff(self, circuit_desc: str) -> WolframResult:
        """Solve circuit using Kirchhoff's laws.

        Args:
            circuit_desc: Description of circuit for KCL/KVL analysis

        Returns:
            WolframResult with node voltages and branch currents
        """
        query = f"Kirchhoff's laws analysis: {circuit_desc}"
        return await self.solve(query)

    async def solve_rc_circuit(
        self, resistance: float, capacitance: float, voltage: float = None
    ) -> WolframResult:
        """Analyze RC circuit time constant and response.

        Args:
            resistance: Resistance in ohms
            capacitance: Capacitance in farads
            voltage: Source voltage (optional)

        Returns:
            WolframResult with time constant and response
        """
        query = f"RC circuit with R={resistance} ohms and C={capacitance} farads"
        if voltage:
            query += f" and V={voltage} volts"
        query += ", calculate time constant and step response"
        return await self.solve(query)

    def _build_circuit_query(self, description: str) -> str:
        """Build Wolfram query from circuit description."""
        desc_lower = description.lower()

        # Check for specific circuit types
        if "ohm" in desc_lower and ("volt" in desc_lower or "amp" in desc_lower):
            # Extract values for Ohm's law
            return f"solve circuit: {description}"
        elif "series" in desc_lower:
            return f"equivalent resistance: {description}"
        elif "parallel" in desc_lower:
            return f"parallel resistance: {description}"
        elif "kirchhoff" in desc_lower or "loop" in desc_lower:
            return f"Kirchhoff analysis: {description}"
        elif "rc" in desc_lower or ("resistor" in desc_lower and "capacitor" in desc_lower):
            return f"RC circuit analysis: {description}"
        else:
            return f"circuit analysis: {description}"

    def _build_vector_query(
        self, operation: str, v1: List[float], v2: Optional[List[float]]
    ) -> str:
        """Build Wolfram query for vector operation."""
        v1_str = f"({', '.join(str(x) for x in v1)})"

        if operation == "magnitude":
            return f"magnitude of vector {v1_str}"
        elif operation == "normalize":
            return f"normalize vector {v1_str}"
        elif v2 is not None:
            v2_str = f"({', '.join(str(x) for x in v2)})"
            if operation == "cross":
                return f"cross product of {v1_str} and {v2_str}"
            elif operation == "dot":
                return f"dot product of {v1_str} and {v2_str}"
            elif operation == "angle":
                return f"angle between vectors {v1_str} and {v2_str}"
            elif operation == "projection":
                return f"projection of {v1_str} onto {v2_str}"

        return f"vector {operation} {v1_str}"

    def _parse_response(self, query: str, data: Dict[str, Any]) -> WolframResult:
        """Parse Wolfram Alpha JSON response."""
        queryresult = data.get("queryresult", {})

        if not queryresult.get("success", False):
            return WolframResult(
                success=False,
                query=query,
                result="",
                steps=[],
                plots=[],
                raw_pods={},
                error=queryresult.get("error", {}).get("msg", "Query failed")
            )

        pods = queryresult.get("pods", [])
        raw_pods = {}
        result_text = ""
        steps = []
        plots = []

        for pod in pods:
            pod_id = pod.get("id", "")
            pod_title = pod.get("title", "")
            raw_pods[pod_id] = pod

            # Extract subpod content
            for subpod in pod.get("subpods", []):
                plaintext = subpod.get("plaintext", "")
                img = subpod.get("img", {})

                # Primary result
                if pod_id in ("Result", "Solution", "DecimalApproximation", "Decimal approximation"):
                    if plaintext:
                        result_text = plaintext

                # Solution steps
                if "step" in pod_title.lower() or pod_id == "StepByStepSolution":
                    if plaintext:
                        steps.append(plaintext)

                # Plots/images
                if img and img.get("src"):
                    if "plot" in pod_title.lower() or "graph" in pod_title.lower():
                        plots.append(img["src"])

        # If no explicit result, try to find one
        if not result_text:
            for pod_id in ("Input interpretation", "Result", "Solution", "Value"):
                if pod_id in raw_pods:
                    for subpod in raw_pods[pod_id].get("subpods", []):
                        if subpod.get("plaintext"):
                            result_text = subpod["plaintext"]
                            break

        return WolframResult(
            success=True,
            query=query,
            result=result_text,
            steps=steps,
            plots=plots,
            raw_pods=raw_pods
        )


# Global instance
wolfram_client = WolframClient()
