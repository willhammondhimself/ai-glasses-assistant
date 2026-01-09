"""
PhysicsEngine: Hybrid SymPy + Wolfram Alpha physics problem solver with step-by-step derivations.

Features:
- Symbolic math: integrals, derivatives, equations (SymPy)
- Physics formula database (kinematics, dynamics, energy, etc.)
- Vector operations (SymPy + Wolfram)
- Circuit analysis via Wolfram Alpha (Ohm's law, Kirchhoff, RC/RL)
- Step-by-step derivation generation for AR overlay
- LaTeX parsing from OCR
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum

import sympy

logger = logging.getLogger(__name__)
from sympy import (
    symbols, Symbol, Eq, solve, simplify, expand, factor,
    diff, integrate, limit, series, sqrt, sin, cos, tan,
    log, ln, exp, pi, oo, I, E, Rational, Float,
    Matrix, det, trace, eye, zeros, ones,
    latex, pretty, init_printing
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
from sympy.vector import CoordSys3D, divergence, curl, gradient


class ProblemType(Enum):
    """Types of physics/math problems."""
    INTEGRAL = "integral"
    DERIVATIVE = "derivative"
    EQUATION = "equation"
    SIMPLIFY = "simplify"
    EVALUATE = "evaluate"
    VECTOR = "vector"
    MATRIX = "matrix"
    PHYSICS_FORMULA = "physics_formula"
    LIMIT = "limit"
    SERIES = "series"
    CIRCUIT = "circuit"  # Wolfram Alpha for circuits
    UNKNOWN = "unknown"


@dataclass
class PhysicsSolution:
    """Result of a physics problem solution."""
    problem: str
    problem_type: ProblemType
    solution: str
    solution_latex: str
    steps: List[str] = field(default_factory=list)
    numeric_value: Optional[float] = None
    method: str = "sympy"
    formula_used: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "problem_type": self.problem_type.value,
            "solution": self.solution,
            "solution_latex": self.solution_latex,
            "steps": self.steps,
            "numeric_value": self.numeric_value,
            "method": self.method,
            "formula_used": self.formula_used,
            "error": self.error
        }


class PhysicsEngine:
    """
    Offline physics and calculus solver using SymPy.

    Supports:
    - Integrals (definite and indefinite)
    - Derivatives
    - Algebraic equations
    - Vector calculus
    - Physics formulas
    - Step-by-step derivations
    """

    # Common physics formulas organized by topic
    PHYSICS_FORMULAS: Dict[str, Dict[str, str]] = {
        "kinematics": {
            "velocity": "v = u + a*t",
            "displacement": "s = u*t + (1/2)*a*t**2",
            "velocity_squared": "v**2 = u**2 + 2*a*s",
            "average_velocity": "v_avg = (u + v) / 2",
            "displacement_avg": "s = v_avg * t",
        },
        "dynamics": {
            "newton_second": "F = m * a",
            "weight": "W = m * g",
            "friction": "f = mu * N",
            "momentum": "p = m * v",
            "impulse": "J = F * t",
            "impulse_momentum": "J = delta_p",
        },
        "energy": {
            "kinetic": "KE = (1/2) * m * v**2",
            "potential_grav": "PE = m * g * h",
            "potential_spring": "PE_spring = (1/2) * k * x**2",
            "work": "W = F * d * cos(theta)",
            "power": "P = W / t",
            "power_velocity": "P = F * v",
        },
        "rotational": {
            "angular_velocity": "omega = theta / t",
            "angular_acceleration": "alpha = delta_omega / t",
            "torque": "tau = r * F * sin(theta)",
            "moment_inertia_point": "I = m * r**2",
            "rotational_ke": "KE_rot = (1/2) * I * omega**2",
            "angular_momentum": "L = I * omega",
        },
        "waves": {
            "wave_speed": "v = f * lambda_w",
            "period": "T = 1 / f",
            "angular_frequency": "omega = 2 * pi * f",
        },
        "thermodynamics": {
            "ideal_gas": "P * V = n * R * T",
            "heat": "Q = m * c * delta_T",
            "work_gas": "W = P * delta_V",
            "efficiency": "eta = W / Q_h",
        },
        "electromagnetism": {
            "coulomb": "F = k * q1 * q2 / r**2",
            "electric_field": "E = F / q",
            "voltage": "V = W / q",
            "ohm": "V = I * R",
            "power_electric": "P = I * V",
            "capacitance": "C = Q / V",
        },
    }

    # Common physics constants
    CONSTANTS: Dict[str, float] = {
        "g": 9.81,           # m/s^2 gravitational acceleration
        "G": 6.674e-11,      # N*m^2/kg^2 gravitational constant
        "c": 299792458,      # m/s speed of light
        "h": 6.626e-34,      # J*s Planck's constant
        "k_B": 1.381e-23,    # J/K Boltzmann constant
        "e": 1.602e-19,      # C elementary charge
        "epsilon_0": 8.854e-12,  # F/m permittivity of free space
        "mu_0": 1.257e-6,    # H/m permeability of free space
        "R": 8.314,          # J/(mol*K) gas constant
        "N_A": 6.022e23,     # 1/mol Avogadro's number
    }

    def __init__(self):
        """Initialize physics engine with SymPy setup and Wolfram client."""
        init_printing(use_unicode=True)

        # Common symbols
        self.x, self.y, self.z, self.t = symbols('x y z t', real=True)
        self.n, self.m, self.k = symbols('n m k', integer=True, positive=True)

        # Physics symbols
        self.v, self.u, self.a, self.s = symbols('v u a s', real=True)
        self.F, self.W, self.E, self.P = symbols('F W E P', real=True)
        self.theta, self.omega, self.alpha = symbols('theta omega alpha', real=True)

        # Transformations for parsing
        self.transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

        # Initialize Wolfram client for advanced problems
        try:
            from backend.physics.wolfram_client import wolfram_client
            self.wolfram = wolfram_client
            logger.info(f"Wolfram Alpha client: {'available' if self.wolfram.available else 'disabled'}")
        except ImportError:
            self.wolfram = None
            logger.warning("Wolfram client not available")

    async def solve(self, problem: str, context: Optional[Dict] = None) -> PhysicsSolution:
        """
        Main entry point for solving physics/math problems.

        Args:
            problem: Natural language or mathematical expression
            context: Optional context (e.g., known values, units)

        Returns:
            PhysicsSolution with solution and steps
        """
        problem = problem.strip()
        context = context or {}

        # Detect problem type
        problem_type = self._detect_problem_type(problem)

        try:
            if problem_type == ProblemType.CIRCUIT:
                return await self._solve_circuit(problem, context)
            elif problem_type == ProblemType.INTEGRAL:
                return self._solve_integral(problem, context)
            elif problem_type == ProblemType.DERIVATIVE:
                return self._solve_derivative(problem, context)
            elif problem_type == ProblemType.EQUATION:
                return self._solve_equation(problem, context)
            elif problem_type == ProblemType.LIMIT:
                return self._solve_limit(problem, context)
            elif problem_type == ProblemType.VECTOR:
                return await self._solve_vector(problem, context)
            elif problem_type == ProblemType.PHYSICS_FORMULA:
                return self._solve_physics(problem, context)
            else:
                return self._solve_expression(problem, context)

        except Exception as e:
            return PhysicsSolution(
                problem=problem,
                problem_type=problem_type,
                solution="",
                solution_latex="",
                steps=[f"Error: {str(e)}"],
                error=str(e)
            )

    def _detect_problem_type(self, problem: str) -> ProblemType:
        """Detect the type of problem from the input."""
        problem_lower = problem.lower()

        # Circuit patterns (check first - routes to Wolfram)
        circuit_keywords = ['circuit', 'resistor', 'capacitor', 'inductor', 'ohm',
                          'kirchhoff', 'series resistance', 'parallel resistance',
                          'rc circuit', 'rl circuit', 'rlc', 'impedance',
                          'time constant', 'voltage divider', 'current divider']
        if any(kw in problem_lower for kw in circuit_keywords):
            return ProblemType.CIRCUIT

        # Integral patterns
        if any(kw in problem_lower for kw in ['integral', 'integrate', '∫']):
            return ProblemType.INTEGRAL
        if re.search(r'\\int|int\s*\(', problem_lower):
            return ProblemType.INTEGRAL

        # Derivative patterns
        if any(kw in problem_lower for kw in ['derivative', 'differentiate', 'd/dx', "d/dt", "'"]):
            return ProblemType.DERIVATIVE
        if re.search(r'\\frac\{d', problem_lower):
            return ProblemType.DERIVATIVE

        # Limit patterns
        if any(kw in problem_lower for kw in ['limit', 'lim', 'approaches']):
            return ProblemType.LIMIT

        # Vector patterns - check for Wolfram-style vector operations
        if any(kw in problem_lower for kw in ['vector', 'curl', 'divergence', 'gradient', 'cross product', 'dot product']):
            return ProblemType.VECTOR

        # Physics formula patterns
        physics_keywords = ['velocity', 'acceleration', 'force', 'energy', 'momentum',
                          'torque', 'power', 'work', 'kinetic', 'potential', 'wave',
                          'frequency', 'voltage', 'current', 'resistance']
        if any(kw in problem_lower for kw in physics_keywords):
            return ProblemType.PHYSICS_FORMULA

        # Equation (has = sign)
        if '=' in problem and not re.search(r'[<>!]=|=+[=>]', problem):
            return ProblemType.EQUATION

        return ProblemType.SIMPLIFY

    def _solve_integral(self, problem: str, context: Dict) -> PhysicsSolution:
        """Solve an integral problem with step-by-step derivation."""
        steps = []

        # Parse the integral expression
        expr_str, var, limits = self._parse_integral(problem)
        steps.append(f"1. Identify integral: ∫ {expr_str} d{var}")

        # Parse expression
        expr = self._parse_expression(expr_str)
        var_sym = self._get_symbol(var)  # Use consistent symbol
        steps.append(f"2. Expression to integrate: {pretty(expr, use_unicode=True)}")

        if limits:
            # Definite integral
            a, b = limits
            steps.append(f"3. Limits: from {a} to {b}")

            # Find antiderivative first
            antideriv = integrate(expr, var_sym)
            steps.append(f"4. Find antiderivative: F({var}) = {pretty(antideriv, use_unicode=True)}")

            # Evaluate at limits
            result = integrate(expr, (var_sym, a, b))
            steps.append(f"5. Apply Fundamental Theorem: F({b}) - F({a})")
            steps.append(f"6. Result: {pretty(result, use_unicode=True)}")

            # Try numeric evaluation
            numeric = None
            try:
                numeric = float(result.evalf())
                steps.append(f"7. Numeric value: {numeric:.6g}")
            except:
                pass
        else:
            # Indefinite integral
            result = integrate(expr, var_sym)
            steps.append(f"3. Apply integration rules")
            steps.append(f"4. Result: {pretty(result, use_unicode=True)} + C")
            numeric = None

        return PhysicsSolution(
            problem=problem,
            problem_type=ProblemType.INTEGRAL,
            solution=str(result),
            solution_latex=latex(result),
            steps=steps,
            numeric_value=numeric
        )

    def _solve_derivative(self, problem: str, context: Dict) -> PhysicsSolution:
        """Solve a derivative problem with step-by-step derivation."""
        steps = []

        # Parse the derivative expression
        expr_str, var, order = self._parse_derivative(problem)
        steps.append(f"1. Identify derivative: d{''.join(['']*order)}/d{var}{''.join(['']*order)} of {expr_str}")

        # Parse expression
        expr = self._parse_expression(expr_str)
        var_sym = self._get_symbol(var)  # Use consistent symbol
        steps.append(f"2. Function to differentiate: f({var}) = {pretty(expr, use_unicode=True)}")

        # Compute derivative step by step
        result = expr
        for i in range(order):
            prev = result
            result = diff(result, var_sym)
            if order == 1:
                steps.append(f"3. Apply differentiation rules")
            else:
                steps.append(f"3.{i+1}. Derivative {i+1}: {pretty(result, use_unicode=True)}")

        steps.append(f"4. Final result: {pretty(result, use_unicode=True)}")

        # Simplify if possible
        simplified = simplify(result)
        if simplified != result:
            steps.append(f"5. Simplified: {pretty(simplified, use_unicode=True)}")
            result = simplified

        return PhysicsSolution(
            problem=problem,
            problem_type=ProblemType.DERIVATIVE,
            solution=str(result),
            solution_latex=latex(result),
            steps=steps
        )

    def _solve_equation(self, problem: str, context: Dict) -> PhysicsSolution:
        """Solve an algebraic equation."""
        steps = []

        # Parse equation
        parts = problem.split('=')
        if len(parts) != 2:
            raise ValueError("Invalid equation format. Expected 'left = right'")

        left_str, right_str = parts[0].strip(), parts[1].strip()
        steps.append(f"1. Equation: {left_str} = {right_str}")

        left = self._parse_expression(left_str)
        right = self._parse_expression(right_str)

        # Create equation
        eq = Eq(left, right)
        steps.append(f"2. Parsed: {pretty(eq, use_unicode=True)}")

        # Find variables
        free_vars = eq.free_symbols
        if not free_vars:
            # No variables - check if equation is true
            is_true = simplify(left - right) == 0
            return PhysicsSolution(
                problem=problem,
                problem_type=ProblemType.EQUATION,
                solution=str(is_true),
                solution_latex=str(is_true),
                steps=steps + [f"3. Equation is: {is_true}"]
            )

        # Solve for variable(s)
        solve_for = context.get('solve_for')
        if solve_for:
            solve_for = Symbol(solve_for)
        else:
            # Default to first variable (prefer x, t, y)
            preferred = ['x', 't', 'y', 'z']
            solve_for = None
            for pref in preferred:
                for var in free_vars:
                    if str(var) == pref:
                        solve_for = var
                        break
                if solve_for:
                    break
            if not solve_for:
                solve_for = list(free_vars)[0]

        steps.append(f"3. Solving for: {solve_for}")

        # Rearrange and solve
        solutions = solve(eq, solve_for)

        if not solutions:
            return PhysicsSolution(
                problem=problem,
                problem_type=ProblemType.EQUATION,
                solution="No solution",
                solution_latex="\\text{No solution}",
                steps=steps + ["4. No solution found"],
                error="No solution found"
            )

        # Format solution
        if len(solutions) == 1:
            steps.append(f"4. Solution: {solve_for} = {pretty(solutions[0], use_unicode=True)}")
            solution_str = f"{solve_for} = {solutions[0]}"
            solution_latex = f"{latex(solve_for)} = {latex(solutions[0])}"
        else:
            steps.append(f"4. Solutions found: {len(solutions)}")
            for i, sol in enumerate(solutions):
                steps.append(f"   {solve_for}_{i+1} = {pretty(sol, use_unicode=True)}")
            solution_str = f"{solve_for} = {solutions}"
            solution_latex = f"{latex(solve_for)} = " + ", ".join(latex(s) for s in solutions)

        return PhysicsSolution(
            problem=problem,
            problem_type=ProblemType.EQUATION,
            solution=solution_str,
            solution_latex=solution_latex,
            steps=steps
        )

    def _solve_limit(self, problem: str, context: Dict) -> PhysicsSolution:
        """Solve a limit problem."""
        steps = []

        # Parse limit expression
        expr_str, var, point, direction = self._parse_limit(problem)

        dir_str = ""
        if direction == "+":
            dir_str = "⁺"
        elif direction == "-":
            dir_str = "⁻"

        steps.append(f"1. Find: lim({var}→{point}{dir_str}) {expr_str}")

        expr = self._parse_expression(expr_str)
        var_sym = self._get_symbol(var)  # Use consistent symbol

        # Parse point (could be infinity)
        if point in ['inf', 'infinity', '∞']:
            point_val = oo
        elif point in ['-inf', '-infinity', '-∞']:
            point_val = -oo
        else:
            point_val = self._parse_expression(point)

        steps.append(f"2. Expression: {pretty(expr, use_unicode=True)}")

        # Compute limit
        if direction == "+":
            result = limit(expr, var_sym, point_val, '+')
        elif direction == "-":
            result = limit(expr, var_sym, point_val, '-')
        else:
            result = limit(expr, var_sym, point_val)

        steps.append(f"3. Evaluate limit")
        steps.append(f"4. Result: {pretty(result, use_unicode=True)}")

        numeric = None
        try:
            numeric = float(result.evalf())
        except:
            pass

        return PhysicsSolution(
            problem=problem,
            problem_type=ProblemType.LIMIT,
            solution=str(result),
            solution_latex=latex(result),
            steps=steps,
            numeric_value=numeric
        )

    async def _solve_vector(self, problem: str, context: Dict) -> PhysicsSolution:
        """Solve vector calculus problems using SymPy or Wolfram."""
        steps = []
        problem_lower = problem.lower()

        # Check for cross/dot product - use Wolfram if available
        if self.wolfram and self.wolfram.available:
            if 'cross' in problem_lower or 'dot' in problem_lower:
                # Try to extract vectors
                vectors = self._extract_vectors(problem)
                if len(vectors) >= 2:
                    operation = 'cross' if 'cross' in problem_lower else 'dot'
                    result = await self.wolfram.solve_vector(operation, vectors[0], vectors[1])
                    if result.success:
                        return PhysicsSolution(
                            problem=problem,
                            problem_type=ProblemType.VECTOR,
                            solution=result.result,
                            solution_latex=result.result,
                            steps=result.steps if result.steps else [f"Computed {operation} product via Wolfram Alpha"],
                            method="wolfram"
                        )

        # Set up coordinate system for SymPy
        N = CoordSys3D('N')

        if 'gradient' in problem_lower or 'grad' in problem_lower:
            # Gradient of scalar field
            expr_str = self._extract_expression(problem)
            expr = self._parse_expression(expr_str)

            steps.append(f"1. Find gradient of: {expr_str}")

            # Compute gradient
            grad = gradient(expr.subs({self.x: N.x, self.y: N.y, self.z: N.z}))

            steps.append(f"2. ∇f = (∂f/∂x)i + (∂f/∂y)j + (∂f/∂z)k")
            steps.append(f"3. Result: {grad}")

            return PhysicsSolution(
                problem=problem,
                problem_type=ProblemType.VECTOR,
                solution=str(grad),
                solution_latex=latex(grad),
                steps=steps
            )

        # Fallback to Wolfram for complex vector operations
        if self.wolfram and self.wolfram.available:
            result = await self.wolfram.solve(problem)
            if result.success:
                return PhysicsSolution(
                    problem=problem,
                    problem_type=ProblemType.VECTOR,
                    solution=result.result,
                    solution_latex=result.result,
                    steps=result.steps if result.steps else ["Computed via Wolfram Alpha"],
                    method="wolfram"
                )

        # Default vector operations
        return PhysicsSolution(
            problem=problem,
            problem_type=ProblemType.VECTOR,
            solution="Vector operation parsed",
            solution_latex="",
            steps=["Vector calculus support in progress"],
            error="Complex vector operation - needs more context"
        )

    async def _solve_circuit(self, problem: str, context: Dict) -> PhysicsSolution:
        """Solve circuit problems using Wolfram Alpha API.

        Handles:
        - Ohm's law (V = IR)
        - Series/parallel resistor combinations
        - Kirchhoff's laws
        - RC/RL circuit analysis
        - Time constants
        """
        steps = []
        problem_lower = problem.lower()

        # Check if Wolfram is available
        if not self.wolfram or not self.wolfram.available:
            return PhysicsSolution(
                problem=problem,
                problem_type=ProblemType.CIRCUIT,
                solution="",
                solution_latex="",
                steps=["Wolfram Alpha API not configured"],
                error="Circuit solver requires WOLFRAM_APP_ID environment variable"
            )

        steps.append("1. Analyzing circuit problem...")

        # Detect circuit type and solve appropriately
        try:
            # Ohm's law detection
            if 'ohm' in problem_lower or ('volt' in problem_lower and ('amp' in problem_lower or 'resist' in problem_lower)):
                values = self._extract_circuit_values(problem)
                result = await self.wolfram.solve_ohms_law(**values)

            # Series/parallel resistors
            elif 'series' in problem_lower or 'parallel' in problem_lower:
                resistors = self._extract_resistor_values(problem)
                config = 'parallel' if 'parallel' in problem_lower else 'series'
                steps.append(f"2. Detected {config} resistor configuration")
                result = await self.wolfram.solve_series_parallel(resistors, config)

            # Kirchhoff's laws
            elif 'kirchhoff' in problem_lower or 'loop' in problem_lower or 'node' in problem_lower:
                steps.append("2. Using Kirchhoff's laws analysis")
                result = await self.wolfram.solve_kirchhoff(problem)

            # RC circuit
            elif 'rc circuit' in problem_lower or ('resistor' in problem_lower and 'capacitor' in problem_lower):
                values = self._extract_rc_values(problem)
                steps.append("2. Analyzing RC circuit")
                result = await self.wolfram.solve_rc_circuit(**values)

            # General circuit query
            else:
                steps.append("2. Sending to Wolfram Alpha")
                result = await self.wolfram.solve_circuit(problem)

            # Process result
            if result.success:
                steps.append(f"3. Solution: {result.result}")
                if result.steps:
                    steps.extend([f"  {s}" for s in result.steps])

                return PhysicsSolution(
                    problem=problem,
                    problem_type=ProblemType.CIRCUIT,
                    solution=result.result,
                    solution_latex=result.result,
                    steps=steps,
                    method="wolfram"
                )
            else:
                return PhysicsSolution(
                    problem=problem,
                    problem_type=ProblemType.CIRCUIT,
                    solution="",
                    solution_latex="",
                    steps=steps + [f"Error: {result.error}"],
                    error=result.error
                )

        except Exception as e:
            logger.error(f"Circuit solver error: {e}")
            return PhysicsSolution(
                problem=problem,
                problem_type=ProblemType.CIRCUIT,
                solution="",
                solution_latex="",
                steps=steps + [f"Error: {str(e)}"],
                error=str(e)
            )

    def _extract_vectors(self, problem: str) -> List[List[float]]:
        """Extract vector components from problem text."""
        vectors = []
        # Look for patterns like (1, 2, 3) or [1, 2, 3] or <1, 2, 3>
        patterns = [
            r'\(([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\)',
            r'\[([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\]',
            r'<([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)>',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, problem)
            for match in matches:
                vectors.append([float(x) for x in match])
        return vectors

    def _extract_circuit_values(self, problem: str) -> Dict[str, float]:
        """Extract voltage, current, resistance from problem text."""
        values = {}
        problem_lower = problem.lower()

        # Voltage patterns
        volt_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:v(?:olt)?s?)\b', problem_lower)
        if volt_match:
            values['voltage'] = float(volt_match.group(1))

        # Current patterns (amps or milliamps)
        amp_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m?a(?:mp)?s?)\b', problem_lower)
        if amp_match:
            current = float(amp_match.group(1))
            if 'ma' in problem_lower or 'milliamp' in problem_lower:
                current /= 1000
            values['current'] = current

        # Resistance patterns (ohms or kilohms)
        ohm_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:k?(?:ohm|Ω)s?)\b', problem_lower)
        if ohm_match:
            resistance = float(ohm_match.group(1))
            if 'kohm' in problem_lower or 'kilohm' in problem_lower:
                resistance *= 1000
            values['resistance'] = resistance

        return values

    def _extract_resistor_values(self, problem: str) -> List[float]:
        """Extract resistor values from problem text."""
        resistors = []
        # Look for patterns like "10 ohm" or "10Ω" or "10 ohms"
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:k)?(?:ohm|Ω)s?', problem.lower())
        for match in matches:
            resistors.append(float(match))
        return resistors if resistors else [0]

    def _extract_rc_values(self, problem: str) -> Dict[str, float]:
        """Extract RC circuit values from problem text."""
        values = {}
        problem_lower = problem.lower()

        # Resistance
        r_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:k)?ohm', problem_lower)
        if r_match:
            values['resistance'] = float(r_match.group(1))
            if 'kohm' in problem_lower:
                values['resistance'] *= 1000

        # Capacitance (farads, microfarads, nanofarads, picofarads)
        c_match = re.search(r'(\d+(?:\.\d+)?)\s*([munp])?f(?:arad)?', problem_lower)
        if c_match:
            cap = float(c_match.group(1))
            prefix = c_match.group(2)
            if prefix == 'm':
                cap *= 1e-3
            elif prefix == 'u' or prefix == 'μ':
                cap *= 1e-6
            elif prefix == 'n':
                cap *= 1e-9
            elif prefix == 'p':
                cap *= 1e-12
            values['capacitance'] = cap

        # Voltage
        v_match = re.search(r'(\d+(?:\.\d+)?)\s*v(?:olt)?', problem_lower)
        if v_match:
            values['voltage'] = float(v_match.group(1))

        return values

    def _solve_physics(self, problem: str, context: Dict) -> PhysicsSolution:
        """Solve physics problems using formula database."""
        steps = []
        problem_lower = problem.lower()

        # Find relevant formula category
        formula_used = None
        for category, formulas in self.PHYSICS_FORMULAS.items():
            for name, formula in formulas.items():
                if name.replace('_', ' ') in problem_lower or any(
                    kw in problem_lower for kw in name.split('_')
                ):
                    formula_used = f"{category}/{name}: {formula}"
                    steps.append(f"1. Relevant formula: {formula}")
                    break
            if formula_used:
                break

        if not formula_used:
            steps.append("1. No specific formula matched - treating as general equation")

        # Extract values from context or problem
        known_values = context.get('values', {})

        # Try to solve as equation
        return self._solve_equation(problem, context)

    def _solve_expression(self, problem: str, context: Dict) -> PhysicsSolution:
        """Simplify or evaluate a mathematical expression."""
        steps = []

        expr = self._parse_expression(problem)
        steps.append(f"1. Parse expression: {pretty(expr, use_unicode=True)}")

        # Check if purely numeric
        if not expr.free_symbols:
            result = expr.evalf()
            steps.append(f"2. Evaluate: {result}")

            return PhysicsSolution(
                problem=problem,
                problem_type=ProblemType.EVALUATE,
                solution=str(result),
                solution_latex=latex(result),
                steps=steps,
                numeric_value=float(result)
            )

        # Simplify
        simplified = simplify(expr)
        steps.append(f"2. Simplify: {pretty(simplified, use_unicode=True)}")

        # Try to factor
        factored = factor(expr)
        if factored != simplified:
            steps.append(f"3. Factored form: {pretty(factored, use_unicode=True)}")

        # Try to expand
        expanded = expand(expr)
        if expanded != simplified:
            steps.append(f"4. Expanded form: {pretty(expanded, use_unicode=True)}")

        return PhysicsSolution(
            problem=problem,
            problem_type=ProblemType.SIMPLIFY,
            solution=str(simplified),
            solution_latex=latex(simplified),
            steps=steps
        )

    def _parse_expression(self, expr_str: str) -> sympy.Expr:
        """Parse a string expression into SymPy."""
        # Clean up common notations
        expr_str = self._clean_expression(expr_str)

        # Local dict for parsing
        local_dict = {
            'x': self.x, 'y': self.y, 'z': self.z, 't': self.t,
            'n': self.n, 'm': self.m, 'k': self.k,
            'pi': pi, 'e': E, 'i': I, 'inf': oo,
            'sin': sin, 'cos': cos, 'tan': tan,
            'log': log, 'ln': ln, 'exp': exp, 'sqrt': sqrt,
        }

        return parse_expr(expr_str, local_dict=local_dict, transformations=self.transformations)

    def _get_symbol(self, var_name: str) -> Symbol:
        """Get the consistent Symbol object for a variable name."""
        symbol_map = {
            'x': self.x, 'y': self.y, 'z': self.z, 't': self.t,
            'n': self.n, 'm': self.m, 'k': self.k,
        }
        return symbol_map.get(var_name, Symbol(var_name))

    def _clean_expression(self, expr: str) -> str:
        """Clean and normalize an expression string."""
        result = expr.strip()

        # LaTeX conversions
        conversions = [
            (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))'),
            (r'\\sqrt\{([^}]+)\}', r'sqrt(\1)'),
            (r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', r'(\2)**(1/(\1))'),
            (r'\^(\d)', r'**\1'),
            (r'\^\{([^}]+)\}', r'**(\1)'),
            (r'\\cdot', '*'),
            (r'\\times', '*'),
            (r'\\div', '/'),
            (r'\\pi', 'pi'),
            (r'\\infty', 'oo'),
            (r'\\sin', 'sin'),
            (r'\\cos', 'cos'),
            (r'\\tan', 'tan'),
            (r'\\log', 'log'),
            (r'\\ln', 'ln'),
            (r'\\exp', 'exp'),
            (r'\\left', ''),
            (r'\\right', ''),
            (r'\{', '('),
            (r'\}', ')'),
            (r'\$', ''),
        ]

        for pattern, replacement in conversions:
            result = re.sub(pattern, replacement, result)

        # Handle implicit multiplication (2x -> 2*x)
        result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)

        return result

    def _parse_integral(self, problem: str) -> tuple:
        """Parse integral notation and return (expression, variable, limits)."""
        problem_lower = problem.lower()

        # Extract variable (default to x)
        var = 'x'
        var_match = re.search(r'd([a-z])', problem_lower)
        if var_match:
            var = var_match.group(1)

        # Check for definite integral limits
        limits = None
        limit_match = re.search(r'from\s+([\d.-]+)\s+to\s+([\d.-]+)', problem_lower)
        if limit_match:
            limits = (float(limit_match.group(1)), float(limit_match.group(2)))
        else:
            # Check bracket notation [a, b]
            bracket_match = re.search(r'\[([\d.-]+),\s*([\d.-]+)\]', problem)
            if bracket_match:
                limits = (float(bracket_match.group(1)), float(bracket_match.group(2)))

        # Extract expression to integrate
        expr = problem
        # Remove integral keywords
        for kw in ['integral of', 'integrate', '∫', r'int\(', 'int ']:
            expr = re.sub(kw, '', expr, flags=re.IGNORECASE)
        # Remove dx, dt, etc.
        expr = re.sub(r'\s*d[a-z]\s*', '', expr)
        # Remove limit specifications
        expr = re.sub(r'from\s+[\d.-]+\s+to\s+[\d.-]+', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\[[\d.-]+,\s*[\d.-]+\]', '', expr)

        return expr.strip(), var, limits

    def _parse_derivative(self, problem: str) -> tuple:
        """Parse derivative notation and return (expression, variable, order)."""
        problem_lower = problem.lower()

        # Default values
        var = 'x'
        order = 1

        # Detect order
        if 'second' in problem_lower or "d^2" in problem_lower or "d²" in problem_lower:
            order = 2
        elif 'third' in problem_lower or "d^3" in problem_lower or "d³" in problem_lower:
            order = 3

        # Extract variable
        var_match = re.search(r'd/d([a-z])', problem_lower)
        if var_match:
            var = var_match.group(1)

        # Extract expression
        expr = problem
        for kw in ['derivative of', 'differentiate', "d/dx", "d/dt", "d/dy"]:
            expr = re.sub(kw, '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'(first|second|third)\s+derivative\s+of', '', expr, flags=re.IGNORECASE)

        return expr.strip(), var, order

    def _parse_limit(self, problem: str) -> tuple:
        """Parse limit notation and return (expression, variable, point, direction)."""
        problem_lower = problem.lower()

        # Defaults
        var = 'x'
        point = '0'
        direction = None

        # Extract variable and point
        limit_match = re.search(r'([a-z])\s*(->|→|approaches)\s*([-\d.]+|inf(?:inity)?|∞)', problem_lower)
        if limit_match:
            var = limit_match.group(1)
            point = limit_match.group(3)

        # Check direction
        if 'from right' in problem_lower or 'from above' in problem_lower or '+' in problem:
            direction = '+'
        elif 'from left' in problem_lower or 'from below' in problem_lower:
            direction = '-'

        # Extract expression
        expr = problem
        expr = re.sub(r'lim(?:it)?\s*\(?', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'as\s+[a-z]\s*(->|→|approaches)\s*[-\d.]+', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'[a-z]\s*(->|→)\s*[-\d.]+', '', expr)

        return expr.strip(), var, point, direction

    def _extract_expression(self, problem: str) -> str:
        """Extract the mathematical expression from a problem description."""
        # Remove common problem phrasing
        expr = problem
        removals = [
            r'find\s+(?:the\s+)?',
            r'calculate\s+(?:the\s+)?',
            r'compute\s+(?:the\s+)?',
            r'gradient\s+of\s+',
            r'divergence\s+of\s+',
            r'curl\s+of\s+',
        ]
        for pattern in removals:
            expr = re.sub(pattern, '', expr, flags=re.IGNORECASE)

        return expr.strip()

    def solve_symbolic(self, equation: str, solve_for: str) -> List[Any]:
        """
        Direct symbolic equation solving.

        Args:
            equation: Equation string (e.g., "x^2 - 4 = 0")
            solve_for: Variable to solve for

        Returns:
            List of solutions
        """
        var = Symbol(solve_for)

        if '=' in equation:
            parts = equation.split('=')
            left = self._parse_expression(parts[0].strip())
            right = self._parse_expression(parts[1].strip())
            eq = Eq(left, right)
        else:
            expr = self._parse_expression(equation)
            eq = Eq(expr, 0)

        return solve(eq, var)

    def get_formula(self, category: str, name: str) -> Optional[str]:
        """Get a physics formula by category and name."""
        if category in self.PHYSICS_FORMULAS:
            return self.PHYSICS_FORMULAS[category].get(name)
        return None

    def list_formulas(self, category: Optional[str] = None) -> Dict:
        """List available physics formulas."""
        if category:
            return {category: self.PHYSICS_FORMULAS.get(category, {})}
        return self.PHYSICS_FORMULAS

    def get_constant(self, name: str) -> Optional[float]:
        """Get a physics constant value."""
        return self.CONSTANTS.get(name)


# Convenience function for quick solving
async def quick_solve(problem: str) -> PhysicsSolution:
    """Quick solve a physics/math problem."""
    engine = PhysicsEngine()
    return await engine.solve(problem)
