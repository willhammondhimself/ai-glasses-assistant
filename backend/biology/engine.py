"""
BiologyEngine: Biology problem solver for genetics and concept explanation.

Features:
- Punnett square generation for monohybrid and dihybrid crosses
- Genotype/phenotype probability calculations
- Biology concept explanations via Claude
- Genetics problem solving
"""

import os
import re
from typing import Optional, Dict, List, Tuple
from itertools import product
import anthropic
from dotenv import load_dotenv

load_dotenv()


class BiologyEngine:
    """Biology assistant with local genetics calculations and Claude explanations."""

    def __init__(self):
        self.client: Optional[anthropic.Anthropic] = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

    def punnett_square(self, parent1: str, parent2: str) -> dict:
        """
        Generate a Punnett square for genetic crosses.

        Args:
            parent1: Genotype of first parent (e.g., "Aa", "AaBb")
            parent2: Genotype of second parent (e.g., "Aa", "AaBb")

        Returns:
            dict with Punnett square grid, genotype ratios, and phenotype ratios
        """
        try:
            # Validate and parse genotypes
            alleles1 = self._parse_genotype(parent1)
            alleles2 = self._parse_genotype(parent2)

            if not alleles1 or not alleles2:
                return {
                    "grid": None,
                    "error": f"Could not parse genotypes. Use format like 'Aa' or 'AaBb'"
                }

            if len(alleles1) != len(alleles2):
                return {
                    "grid": None,
                    "error": "Parents must have the same number of genes"
                }

            # Generate gametes for each parent
            gametes1 = self._generate_gametes(alleles1)
            gametes2 = self._generate_gametes(alleles2)

            # Create the Punnett square grid
            grid = []
            all_offspring = []

            for g2 in gametes2:  # Rows
                row = []
                for g1 in gametes1:  # Columns
                    # Combine gametes to form offspring genotype
                    offspring = self._combine_gametes(g1, g2)
                    row.append(offspring)
                    all_offspring.append(offspring)
                grid.append(row)

            # Calculate genotype ratios
            genotype_counts = {}
            for offspring in all_offspring:
                normalized = self._normalize_genotype(offspring)
                genotype_counts[normalized] = genotype_counts.get(normalized, 0) + 1

            total = len(all_offspring)
            genotype_ratios = {k: f"{v}/{total}" for k, v in genotype_counts.items()}

            # Calculate phenotype ratios (assuming simple dominance)
            phenotype_counts = {}
            for offspring in all_offspring:
                phenotype = self._determine_phenotype(offspring)
                phenotype_counts[phenotype] = phenotype_counts.get(phenotype, 0) + 1

            phenotype_ratios = {k: f"{v}/{total}" for k, v in phenotype_counts.items()}

            # Calculate probabilities
            genotype_probabilities = {k: round(v / total, 4) for k, v in genotype_counts.items()}
            phenotype_probabilities = {k: round(v / total, 4) for k, v in phenotype_counts.items()}

            return {
                "grid": grid,
                "parent1": parent1,
                "parent2": parent2,
                "gametes1": gametes1,
                "gametes2": gametes2,
                "genotype_ratios": genotype_ratios,
                "phenotype_ratios": phenotype_ratios,
                "genotype_probabilities": genotype_probabilities,
                "phenotype_probabilities": phenotype_probabilities,
                "total_offspring": total,
                "error": None
            }

        except Exception as e:
            return {
                "grid": None,
                "error": str(e)
            }

    def calculate_probability(
        self,
        target_genotype: str,
        parent1: str,
        parent2: str
    ) -> dict:
        """
        Calculate the probability of a specific genotype from a cross.

        Args:
            target_genotype: The genotype to find probability for
            parent1: First parent's genotype
            parent2: Second parent's genotype

        Returns:
            dict with probability and explanation
        """
        try:
            # Generate Punnett square
            result = self.punnett_square(parent1, parent2)

            if result.get("error"):
                return result

            # Find the target genotype
            normalized_target = self._normalize_genotype(target_genotype)
            probability = result["genotype_probabilities"].get(normalized_target, 0)

            return {
                "target_genotype": target_genotype,
                "normalized_genotype": normalized_target,
                "probability": probability,
                "percentage": f"{probability * 100:.1f}%",
                "ratio": result["genotype_ratios"].get(normalized_target, "0/0"),
                "parent1": parent1,
                "parent2": parent2,
                "error": None
            }

        except Exception as e:
            return {"probability": None, "error": str(e)}

    def explain_concept(self, concept: str) -> dict:
        """
        Explain a biology concept using Claude.

        Args:
            concept: Biology concept or term to explain

        Returns:
            dict with explanation and key points
        """
        if not self.client:
            return {
                "explanation": None,
                "error": "ANTHROPIC_API_KEY not configured"
            }

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Explain this biology concept clearly and concisely: {concept}

Format your response as:
EXPLANATION: [2-3 sentence overview]
KEY_POINTS:
- [point 1]
- [point 2]
- [point 3]
EXAMPLE: [real-world example if applicable]
RELATED_CONCEPTS: [comma-separated list of related terms]"""
                    }
                ]
            )

            response = message.content[0].text

            # Parse response
            explanation = None
            key_points = []
            example = None
            related = []

            exp_match = re.search(r'EXPLANATION:\s*(.+?)(?=KEY_POINTS:|$)', response, re.DOTALL)
            if exp_match:
                explanation = exp_match.group(1).strip()

            kp_match = re.search(r'KEY_POINTS:\s*(.+?)(?=EXAMPLE:|RELATED|$)', response, re.DOTALL)
            if kp_match:
                kp_text = kp_match.group(1).strip()
                key_points = [line.strip().lstrip('- ') for line in kp_text.split('\n') if line.strip().startswith('-')]

            ex_match = re.search(r'EXAMPLE:\s*(.+?)(?=RELATED|$)', response, re.DOTALL)
            if ex_match:
                example = ex_match.group(1).strip()

            rel_match = re.search(r'RELATED_CONCEPTS:\s*(.+?)$', response, re.DOTALL)
            if rel_match:
                related = [r.strip() for r in rel_match.group(1).split(',') if r.strip()]

            return {
                "concept": concept,
                "explanation": explanation or response,
                "key_points": key_points,
                "example": example,
                "related_concepts": related,
                "error": None
            }

        except Exception as e:
            return {
                "explanation": None,
                "error": f"Claude error: {str(e)}"
            }

    def solve_genetics_problem(self, problem: str) -> dict:
        """
        Solve a genetics word problem using Claude.

        Args:
            problem: Genetics problem description

        Returns:
            dict with solution and explanation
        """
        if not self.client:
            return {
                "solution": None,
                "error": "ANTHROPIC_API_KEY not configured"
            }

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1200,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Solve this genetics problem step by step:

{problem}

Format your response as:
ANSWER: [final answer - genotype ratios, probabilities, or statement]
STEPS:
1. [identify the cross type and alleles]
2. [set up the Punnett square or calculation]
3. [calculate probabilities]
PUNNETT_SQUARE: [if applicable, show the grid]
EXPLANATION: [brief explanation of the result]"""
                    }
                ]
            )

            response = message.content[0].text

            # Parse response
            answer = None
            steps = None
            punnett = None
            explanation = None

            ans_match = re.search(r'ANSWER:\s*(.+?)(?=STEPS:|$)', response, re.DOTALL)
            if ans_match:
                answer = ans_match.group(1).strip()

            steps_match = re.search(r'STEPS:\s*(.+?)(?=PUNNETT|EXPLANATION|$)', response, re.DOTALL)
            if steps_match:
                steps = steps_match.group(1).strip()

            pun_match = re.search(r'PUNNETT_SQUARE:\s*(.+?)(?=EXPLANATION|$)', response, re.DOTALL)
            if pun_match:
                punnett = pun_match.group(1).strip()

            exp_match = re.search(r'EXPLANATION:\s*(.+?)$', response, re.DOTALL)
            if exp_match:
                explanation = exp_match.group(1).strip()

            return {
                "answer": answer or response,
                "steps": steps,
                "punnett_square": punnett,
                "explanation": explanation,
                "method": "claude",
                "error": None
            }

        except Exception as e:
            return {
                "solution": None,
                "error": f"Claude error: {str(e)}"
            }

    def _parse_genotype(self, genotype: str) -> List[Tuple[str, str]]:
        """
        Parse a genotype string into pairs of alleles.

        Examples:
            "Aa" -> [('A', 'a')]
            "AaBb" -> [('A', 'a'), ('B', 'b')]
            "AaBbCc" -> [('A', 'a'), ('B', 'b'), ('C', 'c')]
        """
        genotype = genotype.strip()
        alleles = []

        # Parse pairs of letters
        i = 0
        while i < len(genotype):
            if i + 1 < len(genotype):
                allele1 = genotype[i]
                allele2 = genotype[i + 1]

                # Validate: should be same letter, one uppercase, one lowercase (or both same)
                if allele1.upper() == allele2.upper():
                    alleles.append((allele1, allele2))
                    i += 2
                else:
                    return []
            else:
                return []

        return alleles

    def _generate_gametes(self, alleles: List[Tuple[str, str]]) -> List[str]:
        """
        Generate all possible gametes from a genotype.

        Example:
            [('A', 'a'), ('B', 'b')] -> ['AB', 'Ab', 'aB', 'ab']
        """
        if not alleles:
            return []

        # Get all combinations
        combinations = list(product(*alleles))

        # Join each combination into a gamete string
        return [''.join(combo) for combo in combinations]

    def _combine_gametes(self, gamete1: str, gamete2: str) -> str:
        """
        Combine two gametes to form offspring genotype.

        Example:
            "AB", "ab" -> "AaBb"
        """
        result = []
        for a1, a2 in zip(gamete1, gamete2):
            # Put uppercase (dominant) first
            if a1.isupper():
                result.append(a1 + a2)
            else:
                result.append(a2 + a1)
        return ''.join(result)

    def _normalize_genotype(self, genotype: str) -> str:
        """
        Normalize a genotype to standard form (uppercase first in each pair).

        Example:
            "aABb" -> "AaBb"
        """
        result = []
        for i in range(0, len(genotype), 2):
            pair = genotype[i:i+2]
            if len(pair) == 2:
                if pair[0].isupper():
                    result.append(pair)
                else:
                    result.append(pair[1] + pair[0])
        return ''.join(result)

    def _determine_phenotype(self, genotype: str) -> str:
        """
        Determine the phenotype assuming simple dominance.

        Returns a string describing dominant/recessive for each trait.
        """
        phenotypes = []
        for i in range(0, len(genotype), 2):
            pair = genotype[i:i+2]
            gene_letter = pair[0].upper()

            # If either allele is uppercase (dominant), phenotype is dominant
            if pair[0].isupper() or pair[1].isupper():
                phenotypes.append(f"{gene_letter}-dominant")
            else:
                phenotypes.append(f"{gene_letter}-recessive")

        return ", ".join(phenotypes)
