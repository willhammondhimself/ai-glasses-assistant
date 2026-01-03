"""
ChemistryEngine: Chemistry problem solver with SymPy and Claude fallback.

Features:
- Balance chemical equations using matrix methods
- Stoichiometry calculations
- Molecular weight calculations
- Molarity/concentration calculations
- Claude fallback for complex word problems
"""

import os
import re
from typing import Optional, Dict, List, Tuple
from fractions import Fraction
import sympy
from sympy import symbols, Eq, solve, Matrix, lcm
import anthropic
from dotenv import load_dotenv

load_dotenv()


# Periodic table with atomic weights (most common isotopes)
PERIODIC_TABLE = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81, 'C': 12.01,
    'N': 14.01, 'O': 16.00, 'F': 19.00, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.31,
    'Al': 26.98, 'Si': 28.09, 'P': 30.97, 'S': 32.07, 'Cl': 35.45, 'Ar': 39.95,
    'K': 39.10, 'Ca': 40.08, 'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 52.00,
    'Mn': 54.94, 'Fe': 55.85, 'Co': 58.93, 'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38,
    'Ga': 69.72, 'Ge': 72.63, 'As': 74.92, 'Se': 78.97, 'Br': 79.90, 'Kr': 83.80,
    'Rb': 85.47, 'Sr': 87.62, 'Y': 88.91, 'Zr': 91.22, 'Nb': 92.91, 'Mo': 95.95,
    'Ru': 101.1, 'Rh': 102.9, 'Pd': 106.4, 'Ag': 107.9, 'Cd': 112.4, 'In': 114.8,
    'Sn': 118.7, 'Sb': 121.8, 'Te': 127.6, 'I': 126.9, 'Xe': 131.3, 'Cs': 132.9,
    'Ba': 137.3, 'La': 138.9, 'Ce': 140.1, 'Pr': 140.9, 'Nd': 144.2, 'Sm': 150.4,
    'Eu': 152.0, 'Gd': 157.3, 'Tb': 158.9, 'Dy': 162.5, 'Ho': 164.9, 'Er': 167.3,
    'Tm': 168.9, 'Yb': 173.0, 'Lu': 175.0, 'Hf': 178.5, 'Ta': 180.9, 'W': 183.8,
    'Re': 186.2, 'Os': 190.2, 'Ir': 192.2, 'Pt': 195.1, 'Au': 197.0, 'Hg': 200.6,
    'Tl': 204.4, 'Pb': 207.2, 'Bi': 209.0, 'Po': 209.0, 'At': 210.0, 'Rn': 222.0,
    'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.0, 'Pa': 231.0, 'U': 238.0
}


class ChemistryEngine:
    """Chemistry problem solver with local calculations and Claude fallback."""

    def __init__(self):
        self.client: Optional[anthropic.Anthropic] = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

    def balance_equation(self, equation: str) -> dict:
        """
        Balance a chemical equation.

        Args:
            equation: Chemical equation string (e.g., "H2 + O2 -> H2O")

        Returns:
            dict with balanced equation, coefficients, and method used
        """
        try:
            # Parse the equation
            reactants, products = self._parse_equation(equation)

            if not reactants or not products:
                return {
                    "balanced": None,
                    "coefficients": None,
                    "error": "Could not parse equation. Use format: H2 + O2 -> H2O"
                }

            # Get all unique elements
            all_compounds = reactants + products
            elements = set()
            for compound in all_compounds:
                elements.update(self._parse_formula(compound).keys())

            # Build the matrix for balancing
            n_compounds = len(all_compounds)
            n_elements = len(elements)
            elements = sorted(elements)

            # Create coefficient matrix
            matrix = []
            for element in elements:
                row = []
                for i, compound in enumerate(all_compounds):
                    formula = self._parse_formula(compound)
                    count = formula.get(element, 0)
                    # Products are negative (moving to other side of equation)
                    if i >= len(reactants):
                        count = -count
                    row.append(count)
                matrix.append(row)

            # Solve using SymPy
            M = Matrix(matrix)
            null_space = M.nullspace()

            if not null_space:
                # Try Claude as fallback
                return self._balance_with_claude(equation)

            # Get the solution vector and convert to integers
            solution = null_space[0]

            # Convert to integers by multiplying by LCM of denominators
            denominators = []
            for val in solution:
                if hasattr(val, 'q'):
                    denominators.append(val.q)
                else:
                    frac = Fraction(float(val)).limit_denominator(1000)
                    denominators.append(frac.denominator)

            multiplier = 1
            for d in denominators:
                multiplier = multiplier * d // sympy.gcd(multiplier, d)

            coefficients = [abs(int(val * multiplier)) for val in solution]

            # Ensure all coefficients are positive and simplify
            gcd = coefficients[0]
            for c in coefficients[1:]:
                gcd = sympy.gcd(gcd, c)
            coefficients = [c // gcd for c in coefficients]

            # Build the balanced equation string
            reactant_parts = []
            for i, compound in enumerate(reactants):
                coef = coefficients[i]
                if coef == 1:
                    reactant_parts.append(compound)
                else:
                    reactant_parts.append(f"{coef}{compound}")

            product_parts = []
            for i, compound in enumerate(products):
                coef = coefficients[len(reactants) + i]
                if coef == 1:
                    product_parts.append(compound)
                else:
                    product_parts.append(f"{coef}{compound}")

            balanced = " + ".join(reactant_parts) + " → " + " + ".join(product_parts)

            return {
                "balanced": balanced,
                "coefficients": {
                    "reactants": dict(zip(reactants, coefficients[:len(reactants)])),
                    "products": dict(zip(products, coefficients[len(reactants):]))
                },
                "method": "matrix",
                "error": None
            }

        except Exception as e:
            # Fallback to Claude
            return self._balance_with_claude(equation)

    def calculate_molecular_weight(self, formula: str) -> dict:
        """
        Calculate the molecular weight of a compound.

        Args:
            formula: Chemical formula (e.g., "H2O", "NaCl", "C6H12O6")

        Returns:
            dict with molecular weight, breakdown by element
        """
        try:
            elements = self._parse_formula(formula)

            if not elements:
                return {
                    "molecular_weight": None,
                    "breakdown": None,
                    "error": f"Could not parse formula: {formula}"
                }

            breakdown = {}
            total_weight = 0

            for element, count in elements.items():
                if element not in PERIODIC_TABLE:
                    return {
                        "molecular_weight": None,
                        "breakdown": None,
                        "error": f"Unknown element: {element}"
                    }

                atomic_weight = PERIODIC_TABLE[element]
                contribution = atomic_weight * count
                breakdown[element] = {
                    "count": count,
                    "atomic_weight": atomic_weight,
                    "contribution": round(contribution, 3)
                }
                total_weight += contribution

            return {
                "molecular_weight": round(total_weight, 3),
                "formula": formula,
                "breakdown": breakdown,
                "unit": "g/mol",
                "error": None
            }

        except Exception as e:
            return {
                "molecular_weight": None,
                "breakdown": None,
                "error": str(e)
            }

    def calculate_molarity(self, solute_moles: float, volume_liters: float) -> dict:
        """
        Calculate molarity (concentration).

        Args:
            solute_moles: Amount of solute in moles
            volume_liters: Volume of solution in liters

        Returns:
            dict with molarity and calculation details
        """
        try:
            if volume_liters <= 0:
                return {"molarity": None, "error": "Volume must be positive"}

            molarity = solute_moles / volume_liters

            return {
                "molarity": round(molarity, 6),
                "unit": "M (mol/L)",
                "solute_moles": solute_moles,
                "volume_liters": volume_liters,
                "formula": "M = n/V",
                "error": None
            }

        except Exception as e:
            return {"molarity": None, "error": str(e)}

    def solve_stoichiometry(self, problem: str) -> dict:
        """
        Solve stoichiometry problems.

        Args:
            problem: Stoichiometry problem description

        Returns:
            dict with solution and steps
        """
        # Try to extract simple mass-to-mass or mole calculations
        # For complex problems, use Claude

        # Check for simple patterns
        mass_pattern = r'(\d+\.?\d*)\s*(g|grams?|kg|kilograms?)\s+(?:of\s+)?([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)'
        mole_pattern = r'(\d+\.?\d*)\s*(mol|moles?)\s+(?:of\s+)?([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)'

        # For now, use Claude for stoichiometry word problems
        return self._solve_with_claude(problem, "stoichiometry")

    def solve_chemistry_problem(self, problem: str) -> dict:
        """
        General chemistry problem solver using Claude.

        Args:
            problem: Chemistry problem in any format

        Returns:
            dict with solution and explanation
        """
        return self._solve_with_claude(problem, "general")

    def _parse_equation(self, equation: str) -> Tuple[List[str], List[str]]:
        """Parse a chemical equation into reactants and products."""
        # Normalize arrow symbols
        equation = equation.replace("→", "->").replace("⟶", "->").replace("=", "->")

        # Split into reactants and products
        parts = equation.split("->")
        if len(parts) != 2:
            return [], []

        reactants_str, products_str = parts

        # Parse compounds (split by +)
        reactants = [c.strip() for c in reactants_str.split("+") if c.strip()]
        products = [c.strip() for c in products_str.split("+") if c.strip()]

        # Remove any leading coefficients
        reactants = [re.sub(r'^\d+', '', r).strip() for r in reactants]
        products = [re.sub(r'^\d+', '', p).strip() for p in products]

        return reactants, products

    def _parse_formula(self, formula: str) -> Dict[str, int]:
        """
        Parse a chemical formula into element counts.

        Examples:
            "H2O" -> {"H": 2, "O": 1}
            "Ca(OH)2" -> {"Ca": 1, "O": 2, "H": 2}
        """
        elements = {}

        # Handle parentheses first
        while '(' in formula:
            # Find innermost parentheses
            match = re.search(r'\(([^()]+)\)(\d*)', formula)
            if not match:
                break

            group_formula = match.group(1)
            multiplier = int(match.group(2)) if match.group(2) else 1

            # Parse the group
            group_elements = self._parse_formula(group_formula)

            # Replace the parenthetical expression with expanded form
            expanded = ''.join(f"{el}{count * multiplier}" for el, count in group_elements.items())
            formula = formula[:match.start()] + expanded + formula[match.end():]

        # Parse remaining formula: element symbols followed by optional numbers
        pattern = r'([A-Z][a-z]?)(\d*)'
        for match in re.finditer(pattern, formula):
            element = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1

            if element:  # Skip empty matches
                elements[element] = elements.get(element, 0) + count

        return elements

    def _balance_with_claude(self, equation: str) -> dict:
        """Use Claude to balance a chemical equation."""
        if not self.client:
            return {
                "balanced": None,
                "coefficients": None,
                "error": "ANTHROPIC_API_KEY not configured and local balancing failed"
            }

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Balance this chemical equation: {equation}

Provide the balanced equation in this exact format:
BALANCED: [balanced equation with coefficients]

Use → for the arrow. Include only the balanced equation, no explanation."""
                    }
                ]
            )

            response = message.content[0].text
            balanced_match = re.search(r'BALANCED:\s*(.+)', response)
            balanced = balanced_match.group(1).strip() if balanced_match else response.strip()

            return {
                "balanced": balanced,
                "coefficients": None,
                "method": "claude",
                "error": None
            }

        except Exception as e:
            return {
                "balanced": None,
                "coefficients": None,
                "error": f"Balancing failed: {str(e)}"
            }

    def _solve_with_claude(self, problem: str, problem_type: str = "general") -> dict:
        """Use Claude to solve a chemistry problem."""
        if not self.client:
            return {
                "solution": None,
                "steps": None,
                "error": "ANTHROPIC_API_KEY not configured"
            }

        try:
            prompt = f"""Solve this {problem_type} chemistry problem step by step:

{problem}

Format your response as:
ANSWER: [final numerical answer with units, or statement]
STEPS:
1. [step 1]
2. [step 2]
..."""

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text

            # Parse response
            answer = None
            steps = None

            answer_match = re.search(r'ANSWER:\s*(.+?)(?=STEPS:|$)', response, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()

            steps_match = re.search(r'STEPS:\s*(.+?)$', response, re.DOTALL)
            if steps_match:
                steps = steps_match.group(1).strip()

            return {
                "solution": answer or response,
                "steps": steps,
                "method": "claude",
                "error": None
            }

        except Exception as e:
            return {
                "solution": None,
                "steps": None,
                "error": f"Claude error: {str(e)}"
            }
