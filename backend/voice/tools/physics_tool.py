"""Voice tool for physics and math problem solving.

Includes circuit analysis via Wolfram Alpha API.
"""
import logging
import re
from typing import Optional

from .base import VoiceTool, VoiceToolResult
from backend.physics.engine import PhysicsEngine, PhysicsSolution

logger = logging.getLogger(__name__)


class PhysicsVoiceTool(VoiceTool):
    """Voice tool for physics and calculus problem solving.

    Handles voice commands like:
    - "Solve integral x squared" - Indefinite integrals
    - "Integrate x cubed from 0 to 5" - Definite integrals
    - "Derivative of sine x" - Derivatives
    - "Solve x squared minus 4 equals 0" - Algebraic equations
    - "Toggle physics HUD" - Activate physics overlay
    - "Show physics mode" - Display physics HUD

    Physics problems:
    - "What's the formula for kinetic energy?" - Formula lookup
    - "Calculate velocity with acceleration 10 time 5" - Physics calculations
    - "Force equals mass times acceleration" - Formula solving

    Circuit analysis (via Wolfram Alpha):
    - "Solve circuit 5 volts 10 ohms" - Ohm's law
    - "3 resistors in series 10 20 30 ohms" - Series resistance
    - "Parallel resistors 100 and 200 ohms" - Parallel resistance
    - "RC circuit 1 kohm 10 microfarad" - Time constant analysis
    - "Kirchhoff loop analysis" - Circuit mesh analysis
    - "Current through 5 volt 100 ohm circuit" - V=IR calculations

    Vector operations (via Wolfram Alpha):
    - "Cross product (1,2,3) and (4,5,6)" - Vector cross product
    - "Dot product of vectors 1 2 3 and 4 5 6" - Dot product

    HUD features:
    - "Physics mode on" - Activate camera OCR for equations
    - "Show step by step" - Display derivation steps
    - "Graph sine x" - Plot functions
    """

    name = "physics"
    description = "Physics and calculus problem solver with circuit analysis"

    # Keywords for activation
    keywords = [
        # Calculus operations
        r'\bintegral\b',
        r'\bintegrate\b',
        r'\bderivative\b',
        r'\bdifferentiate\b',
        r'\bd/dx\b',
        r'\bd/dt\b',
        r'\blimit\b',
        r'\blim\b',
        # Math operations
        r'\bsolve\b',
        r'\bsimplify\b',
        r'\bfactor\b',
        r'\bexpand\b',
        r'\bevaluate\b',
        r'\bcalculate\b',
        # Physics terms
        r'\bphysics\b',
        r'\bformula\b',
        r'\bequation\b',
        r'\bkinetic\s+energy\b',
        r'\bpotential\s+energy\b',
        r'\bvelocity\b',
        r'\bacceleration\b',
        r'\bmomentum\b',
        r'\bforce\b',
        r'\btorque\b',
        r'\bfriction\b',
        # Circuit analysis (Wolfram Alpha)
        r'\bcircuit\b',
        r'\bresistor\b',
        r'\bcapacitor\b',
        r'\binductor\b',
        r'\bohm(?:s)?\b',
        r'\bkirchhoff\b',
        r'\bvolt(?:age|s)?\b',
        r'\bamp(?:s|ere)?\b',
        r'\bcurrent\b.*\b(?:through|circuit)\b',
        r'\bseries\s+(?:resistance|resistor)\b',
        r'\bparallel\s+(?:resistance|resistor)\b',
        r'\brc\s+circuit\b',
        r'\brl\s+circuit\b',
        r'\btime\s+constant\b',
        r'\bimpedance\b',
        # Vector operations
        r'\bcross\s+product\b',
        r'\bdot\s+product\b',
        r'\bvector\b',
        # Math terms
        r'\bx\s*(?:squared|cubed)\b',
        r'\bsine\b',
        r'\bcosine\b',
        r'\btangent\b',
        r'\blogarithm\b',
        r'\bexponential\b',
        r'\bsqrt\b',
        r'\bsquare\s+root\b',
        # HUD commands
        r'\bphysics\s+(?:hud|mode|overlay)\b',
        r'\b(?:show|toggle|enable|disable)\s+physics\b',
        r'\bgraph\s+\w+',
        r'\bplot\s+\w+',
        r'\bstep\s+by\s+step\b',
    ]

    priority = 25  # Higher priority for educational content

    def __init__(self):
        self.engine = PhysicsEngine()

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute physics-related voice command."""
        query_lower = query.lower()

        try:
            # Check for HUD control commands
            if self._is_hud_command(query_lower):
                return await self._handle_hud_command(query_lower)

            # Check for graph/plot commands
            if self._is_graph_command(query_lower):
                return await self._handle_graph_command(query_lower)

            # Check for formula lookup
            if 'formula' in query_lower or 'what is' in query_lower:
                formula = self._extract_formula_request(query_lower)
                if formula:
                    return await self._handle_formula_lookup(formula)

            # Main path: solve the math/physics problem
            return await self._solve_problem(query)

        except Exception as e:
            logger.error(f"Physics tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble with that math problem.",
                data={"error": str(e)}
            )

    def _is_hud_command(self, query: str) -> bool:
        """Check if this is a HUD control command."""
        hud_patterns = [
            r'physics\s+(hud|mode|overlay)',
            r'(show|toggle|enable|disable|start|stop)\s+physics',
            r'(open|close)\s+physics',
            r'step\s+by\s+step\s+(on|off|mode)',
        ]
        return any(re.search(pattern, query) for pattern in hud_patterns)

    def _is_graph_command(self, query: str) -> bool:
        """Check if this is a graph/plot command."""
        return bool(re.search(r'(graph|plot)\s+\w+', query))

    async def _handle_hud_command(self, query: str) -> VoiceToolResult:
        """Handle physics HUD control commands."""

        # Enable/disable physics mode
        if re.search(r'(enable|on|start|show|open)\s*.*physics', query) or \
           re.search(r'physics\s*(hud|mode|overlay)\s*(on)?', query):
            if 'off' in query or 'disable' in query or 'close' in query:
                return VoiceToolResult(
                    success=True,
                    message="Physics mode disabled. Equation OCR overlay hidden.",
                    data={
                        "action": "hud_control",
                        "command": "physics_mode",
                        "hud_message": {"type": "control", "action": "physics_mode", "enabled": False}
                    }
                )
            return VoiceToolResult(
                success=True,
                message="Physics mode enabled. Point your camera at an equation to solve it. Say 'solve integral x squared' for voice input.",
                data={
                    "action": "start_hud",
                    "hud_url": "/hud/physics.html",
                    "stream_endpoint": "/ws/physics-stream",
                    "hud_message": {"type": "control", "action": "physics_mode", "enabled": True}
                }
            )

        # Step by step mode
        if 'step' in query:
            enabled = 'off' not in query and 'disable' not in query
            return VoiceToolResult(
                success=True,
                message=f"Step-by-step mode {'enabled' if enabled else 'disabled'}.",
                data={
                    "action": "hud_control",
                    "command": "step_mode",
                    "hud_message": {"type": "control", "action": "step_mode", "enabled": enabled}
                }
            )

        # Default HUD response
        return VoiceToolResult(
            success=True,
            message="Open the physics HUD at hud slash physics dot html. Point your camera at equations for instant solving.",
            data={
                "action": "info",
                "hud_url": "/hud/physics.html"
            }
        )

    async def _handle_graph_command(self, query: str) -> VoiceToolResult:
        """Handle graph/plot commands."""
        # Extract function to graph
        match = re.search(r'(?:graph|plot)\s+(.+)', query)
        if match:
            func_str = match.group(1).strip()
            func_str = self._convert_spoken_to_math(func_str)

            return VoiceToolResult(
                success=True,
                message=f"Graphing {func_str}. The plot will appear in the physics HUD.",
                data={
                    "action": "graph",
                    "function": func_str,
                    "hud_message": {"type": "graph", "function": func_str}
                }
            )

        return VoiceToolResult(
            success=False,
            message="Please specify a function to graph. For example, 'graph sine x' or 'plot x squared'.",
            data={}
        )

    def _extract_formula_request(self, query: str) -> Optional[str]:
        """Extract what formula is being requested."""
        patterns = [
            r'formula\s+for\s+(\w+(?:\s+\w+)?)',
            r'what\s+is\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+formula',
            r'(\w+(?:\s+\w+)?)\s+equation',
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)
        return None

    async def _handle_formula_lookup(self, formula_name: str) -> VoiceToolResult:
        """Look up a physics formula."""
        formula_name = formula_name.lower().replace(' ', '_')

        # Search through formula database
        for category, formulas in self.engine.PHYSICS_FORMULAS.items():
            for name, formula in formulas.items():
                if formula_name in name or name in formula_name:
                    return VoiceToolResult(
                        success=True,
                        message=f"The {name.replace('_', ' ')} formula is: {formula}",
                        data={
                            "action": "formula_lookup",
                            "category": category,
                            "name": name,
                            "formula": formula
                        }
                    )

        # Special common formulas
        common_formulas = {
            'kinetic': ("kinetic energy", "KE = (1/2) * m * v^2"),
            'potential': ("gravitational potential energy", "PE = m * g * h"),
            'force': ("Newton's second law", "F = m * a"),
            'momentum': ("momentum", "p = m * v"),
            'work': ("work", "W = F * d * cos(theta)"),
            'power': ("power", "P = W / t"),
            'velocity': ("velocity from acceleration", "v = u + a*t"),
            'ohm': ("Ohm's law", "V = I * R"),
        }

        for key, (name, formula) in common_formulas.items():
            if key in formula_name:
                return VoiceToolResult(
                    success=True,
                    message=f"The {name} formula is: {formula}",
                    data={
                        "action": "formula_lookup",
                        "name": name,
                        "formula": formula
                    }
                )

        return VoiceToolResult(
            success=False,
            message=f"I don't have a formula for '{formula_name.replace('_', ' ')}'. Try asking about kinetic energy, momentum, force, or other physics concepts.",
            data={}
        )

    async def _solve_problem(self, query: str) -> VoiceToolResult:
        """Solve a math or physics problem."""
        # Convert spoken math to symbolic notation
        problem = self._convert_spoken_to_math(query)

        # Solve using physics engine
        solution = await self.engine.solve(problem)

        if solution.error:
            return VoiceToolResult(
                success=False,
                message=f"I couldn't solve that: {solution.error}",
                data={"error": solution.error}
            )

        # Format response for voice
        response = self._format_solution_for_voice(solution)

        return VoiceToolResult(
            success=True,
            message=response,
            data={
                "action": "solve",
                "problem": solution.problem,
                "solution": solution.solution,
                "solution_latex": solution.solution_latex,
                "steps": solution.steps,
                "numeric_value": solution.numeric_value,
                "method": solution.method,
                "hud_message": {
                    "type": "solution",
                    "problem": solution.problem,
                    "solution": solution.solution,
                    "latex": solution.solution_latex,
                    "steps": solution.steps
                }
            }
        )

    def _convert_spoken_to_math(self, text: str) -> str:
        """Convert spoken math expressions to symbolic notation."""
        result = text.lower()

        # Strip command prefixes
        prefixes_to_strip = [
            r'^solve\s+',
            r'^calculate\s+',
            r'^compute\s+',
            r'^find\s+(?:the\s+)?',
            r'^what\s+is\s+(?:the\s+)?',
            r'^evaluate\s+',
        ]
        for prefix in prefixes_to_strip:
            result = re.sub(prefix, '', result)

        # Math operations
        conversions = [
            # Powers
            (r'x\s*squared', 'x^2'),
            (r'x\s*cubed', 'x^3'),
            (r'(\w)\s*squared', r'\1^2'),
            (r'(\w)\s*cubed', r'\1^3'),
            (r'(\w)\s*to\s*the\s*(\w+)', lambda m: f"{m.group(1)}^{self._word_to_num(m.group(2))}"),
            (r'square\s*root\s*(?:of\s*)?', 'sqrt('),
            # Trig
            (r'\bsine\b', 'sin'),
            (r'\bcosine\b', 'cos'),
            (r'\btangent\b', 'tan'),
            (r'\barcsin(?:e)?\b', 'asin'),
            (r'\barccos(?:ine)?\b', 'acos'),
            (r'\barctan(?:gent)?\b', 'atan'),
            # Constants
            (r'\bpi\b', 'pi'),
            (r'\be\b', 'E'),
            # Operations
            (r'\btimes\b', '*'),
            (r'\bmultiplied\s*by\b', '*'),
            (r'\bdivided\s*by\b', '/'),
            (r'\bover\b', '/'),
            (r'\bplus\b', '+'),
            (r'\bminus\b', '-'),
            (r'\bequals?\b', '='),
            # Calculus
            (r'\bintegral\s*(?:of\s*)?', 'integral of '),
            (r'\bderivative\s*(?:of\s*)?', 'derivative of '),
            (r'\bwith\s*respect\s*to\s*', 'd'),
            (r'\bfrom\s*(\d+)\s*to\s*(\d+)', r'from \1 to \2'),
            # Circuit units (preserve for Wolfram)
            (r'\bkilohms?\b', 'kohm'),
            (r'\bmilliamps?\b', 'mA'),
            (r'\bmicrofarads?\b', 'uF'),
            (r'\bnanofarads?\b', 'nF'),
            (r'\bpicofarads?\b', 'pF'),
        ]

        for pattern, replacement in conversions:
            if callable(replacement):
                result = re.sub(pattern, replacement, result)
            else:
                result = re.sub(pattern, replacement, result)

        # Close any unclosed sqrt parentheses
        if 'sqrt(' in result and result.count('(') > result.count(')'):
            result += ')'

        return result.strip()

    def _word_to_num(self, word: str) -> str:
        """Convert word number to digit."""
        word_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
        }
        return word_map.get(word.lower(), word)

    def _format_solution_for_voice(self, solution: PhysicsSolution) -> str:
        """Format solution for voice output."""
        problem_type = solution.problem_type.value

        if problem_type == 'integral':
            if solution.numeric_value is not None:
                return f"The integral equals {solution.numeric_value:.4g}"
            return f"The integral is {self._math_to_spoken(solution.solution)}"

        elif problem_type == 'derivative':
            return f"The derivative is {self._math_to_spoken(solution.solution)}"

        elif problem_type == 'equation':
            return f"The solution is {self._math_to_spoken(solution.solution)}"

        elif problem_type == 'limit':
            return f"The limit is {self._math_to_spoken(solution.solution)}"

        elif problem_type == 'circuit':
            # Circuit solutions from Wolfram Alpha
            if solution.solution:
                return f"Circuit analysis: {solution.solution}"
            elif solution.error:
                return f"Circuit solver error: {solution.error}"
            return "Circuit analysis complete. Check the HUD for details."

        elif problem_type == 'vector':
            if solution.solution:
                return f"Vector result: {self._math_to_spoken(solution.solution)}"
            return "Vector calculation complete."

        elif solution.numeric_value is not None:
            return f"The answer is {solution.numeric_value:.4g}"

        return f"The result is {self._math_to_spoken(solution.solution)}"

    def _math_to_spoken(self, expr: str) -> str:
        """Convert mathematical notation to spoken form."""
        result = str(expr)

        # Common conversions
        conversions = [
            (r'\*\*', ' to the power of '),
            (r'\*', ' times '),
            (r'/', ' divided by '),
            (r'sqrt\(([^)]+)\)', r'square root of \1'),
            (r'sin\(', 'sine of '),
            (r'cos\(', 'cosine of '),
            (r'tan\(', 'tangent of '),
            (r'\(', ''),
            (r'\)', ''),
            (r'pi', 'pi'),
            (r'x\^2', 'x squared'),
            (r'x\^3', 'x cubed'),
        ]

        for pattern, replacement in conversions:
            result = re.sub(pattern, replacement, result)

        return result.strip()
