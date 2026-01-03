"""
StatisticsEngine: Statistical analysis using scipy/numpy with Claude for interpretation.

Features:
- Descriptive statistics (mean, median, mode, std, quartiles)
- Hypothesis testing (t-test, chi-square, ANOVA)
- Correlation and regression analysis
- Probability calculations
- Concept explanations via Claude
"""

import os
import re
from typing import Optional, Dict, List, Any, Union
import numpy as np
from scipy import stats
import anthropic
from dotenv import load_dotenv

load_dotenv()


class StatisticsEngine:
    """Statistics assistant with scipy calculations and Claude explanations."""

    def __init__(self):
        self.client: Optional[anthropic.Anthropic] = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

    def descriptive_stats(self, data: List[float]) -> dict:
        """
        Calculate comprehensive descriptive statistics.

        Args:
            data: List of numerical values

        Returns:
            dict with all descriptive statistics
        """
        try:
            if not data:
                return {"error": "Data list is empty"}

            arr = np.array(data, dtype=float)

            # Remove NaN values
            arr = arr[~np.isnan(arr)]

            if len(arr) == 0:
                return {"error": "No valid numerical data"}

            # Calculate statistics
            result = {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "mode": self._calculate_mode(arr),
                "std_dev": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0,
                "variance": float(np.var(arr, ddof=1)) if len(arr) > 1 else 0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "range": float(np.max(arr) - np.min(arr)),
                "sum": float(np.sum(arr)),
            }

            # Quartiles and IQR
            q1, q2, q3 = np.percentile(arr, [25, 50, 75])
            result["q1"] = float(q1)
            result["q2"] = float(q2)  # Same as median
            result["q3"] = float(q3)
            result["iqr"] = float(q3 - q1)

            # Skewness and kurtosis (if enough data)
            if len(arr) >= 3:
                result["skewness"] = float(stats.skew(arr))
            if len(arr) >= 4:
                result["kurtosis"] = float(stats.kurtosis(arr))

            # Standard error of the mean
            result["sem"] = float(stats.sem(arr)) if len(arr) > 1 else 0

            # Round all values
            for key in result:
                if isinstance(result[key], float):
                    result[key] = round(result[key], 6)

            result["error"] = None
            return result

        except Exception as e:
            return {"error": str(e)}

    def hypothesis_test(self, test_type: str, data: Dict[str, Any]) -> dict:
        """
        Perform hypothesis testing.

        Args:
            test_type: Type of test ('t_test', 'paired_t', 'chi_square', 'anova', 'z_test')
            data: Test-specific data (varies by test type)

        Returns:
            dict with test statistic, p-value, and interpretation
        """
        try:
            test_type = test_type.lower().replace('-', '_').replace(' ', '_')

            if test_type == 't_test' or test_type == 'independent_t':
                return self._t_test(data)
            elif test_type == 'paired_t' or test_type == 'paired_t_test':
                return self._paired_t_test(data)
            elif test_type == 'one_sample_t':
                return self._one_sample_t_test(data)
            elif test_type == 'chi_square' or test_type == 'chi2':
                return self._chi_square_test(data)
            elif test_type == 'anova' or test_type == 'one_way_anova':
                return self._anova_test(data)
            elif test_type == 'z_test':
                return self._z_test(data)
            elif test_type == 'mann_whitney' or test_type == 'mann_whitney_u':
                return self._mann_whitney_test(data)
            else:
                return {"error": f"Unknown test type: {test_type}. Supported: t_test, paired_t, one_sample_t, chi_square, anova, z_test, mann_whitney"}

        except Exception as e:
            return {"error": str(e)}

    def correlation(self, x: List[float], y: List[float]) -> dict:
        """
        Calculate correlation between two variables.

        Args:
            x: First variable data
            y: Second variable data

        Returns:
            dict with Pearson and Spearman correlations
        """
        try:
            if len(x) != len(y):
                return {"error": "X and Y must have the same length"}

            if len(x) < 3:
                return {"error": "Need at least 3 data points for correlation"}

            x_arr = np.array(x, dtype=float)
            y_arr = np.array(y, dtype=float)

            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(x_arr, y_arr)

            # Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(x_arr, y_arr)

            # Interpretation
            def interpret_r(r):
                abs_r = abs(r)
                if abs_r < 0.1:
                    strength = "negligible"
                elif abs_r < 0.3:
                    strength = "weak"
                elif abs_r < 0.5:
                    strength = "moderate"
                elif abs_r < 0.7:
                    strength = "strong"
                else:
                    strength = "very strong"
                direction = "positive" if r > 0 else "negative"
                return f"{strength} {direction}"

            return {
                "pearson": {
                    "r": round(float(pearson_r), 6),
                    "r_squared": round(float(pearson_r ** 2), 6),
                    "p_value": round(float(pearson_p), 6),
                    "interpretation": interpret_r(pearson_r)
                },
                "spearman": {
                    "rho": round(float(spearman_r), 6),
                    "p_value": round(float(spearman_p), 6),
                    "interpretation": interpret_r(spearman_r)
                },
                "n": len(x),
                "error": None
            }

        except Exception as e:
            return {"error": str(e)}

    def regression(self, x: List[float], y: List[float]) -> dict:
        """
        Perform simple linear regression.

        Args:
            x: Independent variable data
            y: Dependent variable data

        Returns:
            dict with regression coefficients and statistics
        """
        try:
            if len(x) != len(y):
                return {"error": "X and Y must have the same length"}

            if len(x) < 3:
                return {"error": "Need at least 3 data points for regression"}

            x_arr = np.array(x, dtype=float)
            y_arr = np.array(y, dtype=float)

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_arr, y_arr)

            # Calculate predictions and residuals
            y_pred = slope * x_arr + intercept
            residuals = y_arr - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)

            # Confidence interval for slope (95%)
            n = len(x)
            t_crit = stats.t.ppf(0.975, n - 2)
            slope_ci = (slope - t_crit * std_err, slope + t_crit * std_err)

            return {
                "slope": round(float(slope), 6),
                "intercept": round(float(intercept), 6),
                "r_squared": round(float(r_value ** 2), 6),
                "r": round(float(r_value), 6),
                "p_value": round(float(p_value), 6),
                "std_error": round(float(std_err), 6),
                "equation": f"y = {slope:.4f}x + {intercept:.4f}",
                "slope_95_ci": [round(slope_ci[0], 6), round(slope_ci[1], 6)],
                "n": n,
                "error": None
            }

        except Exception as e:
            return {"error": str(e)}

    def probability_calculation(self, problem: str) -> dict:
        """
        Solve probability problems using Claude.

        Args:
            problem: Probability problem description

        Returns:
            dict with solution and explanation
        """
        if not self.client:
            return {"solution": None, "error": "ANTHROPIC_API_KEY not configured"}

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Solve this probability problem step by step:

{problem}

Format your response as:
ANSWER: [final probability as decimal and percentage]
FORMULA: [probability formula used]
STEPS:
1. [step 1]
2. [step 2]
3. [step 3]"""
                    }
                ]
            )

            response = message.content[0].text

            # Parse response
            answer = None
            formula = None
            steps = None

            ans_match = re.search(r'ANSWER:\s*(.+?)(?=FORMULA:|$)', response, re.DOTALL)
            if ans_match:
                answer = ans_match.group(1).strip()

            form_match = re.search(r'FORMULA:\s*(.+?)(?=STEPS:|$)', response, re.DOTALL)
            if form_match:
                formula = form_match.group(1).strip()

            steps_match = re.search(r'STEPS:\s*(.+?)$', response, re.DOTALL)
            if steps_match:
                steps = steps_match.group(1).strip()

            return {
                "answer": answer or response,
                "formula": formula,
                "steps": steps,
                "method": "claude",
                "error": None
            }

        except Exception as e:
            return {"solution": None, "error": str(e)}

    def explain_stats_concept(self, concept: str) -> dict:
        """
        Explain a statistical concept using Claude.

        Args:
            concept: Statistical concept or term

        Returns:
            dict with explanation
        """
        if not self.client:
            return {"explanation": None, "error": "ANTHROPIC_API_KEY not configured"}

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Explain this statistics concept clearly: {concept}

Format your response as:
DEFINITION: [1-2 sentence definition]
EXPLANATION: [detailed but concise explanation]
FORMULA: [mathematical formula if applicable, or "N/A"]
EXAMPLE: [simple numerical example]
WHEN_TO_USE: [when this concept is useful]"""
                    }
                ]
            )

            response = message.content[0].text

            # Parse response sections
            definition = None
            explanation = None
            formula = None
            example = None
            when_to_use = None

            def_match = re.search(r'DEFINITION:\s*(.+?)(?=EXPLANATION:|$)', response, re.DOTALL)
            if def_match:
                definition = def_match.group(1).strip()

            exp_match = re.search(r'EXPLANATION:\s*(.+?)(?=FORMULA:|$)', response, re.DOTALL)
            if exp_match:
                explanation = exp_match.group(1).strip()

            form_match = re.search(r'FORMULA:\s*(.+?)(?=EXAMPLE:|$)', response, re.DOTALL)
            if form_match:
                formula = form_match.group(1).strip()

            ex_match = re.search(r'EXAMPLE:\s*(.+?)(?=WHEN_TO_USE:|$)', response, re.DOTALL)
            if ex_match:
                example = ex_match.group(1).strip()

            when_match = re.search(r'WHEN_TO_USE:\s*(.+?)$', response, re.DOTALL)
            if when_match:
                when_to_use = when_match.group(1).strip()

            return {
                "concept": concept,
                "definition": definition,
                "explanation": explanation or response,
                "formula": formula,
                "example": example,
                "when_to_use": when_to_use,
                "error": None
            }

        except Exception as e:
            return {"explanation": None, "error": str(e)}

    def _calculate_mode(self, arr: np.ndarray) -> Union[float, List[float], None]:
        """Calculate mode(s) of the data."""
        try:
            mode_result = stats.mode(arr, keepdims=True)
            mode_val = float(mode_result.mode[0])
            count = int(mode_result.count[0])

            # Check if it's actually a mode (appears more than once)
            if count > 1:
                return mode_val
            else:
                return None  # No mode
        except Exception:
            return None

    def _t_test(self, data: Dict) -> dict:
        """Independent samples t-test."""
        group1 = np.array(data.get('group1', []), dtype=float)
        group2 = np.array(data.get('group2', []), dtype=float)
        alpha = data.get('alpha', 0.05)

        if len(group1) < 2 or len(group2) < 2:
            return {"error": "Each group needs at least 2 data points"}

        t_stat, p_value = stats.ttest_ind(group1, group2)

        return {
            "test": "Independent samples t-test",
            "t_statistic": round(float(t_stat), 6),
            "p_value": round(float(p_value), 6),
            "alpha": alpha,
            "significant": p_value < alpha,
            "interpretation": f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis at α={alpha}",
            "group1_mean": round(float(np.mean(group1)), 6),
            "group2_mean": round(float(np.mean(group2)), 6),
            "effect_size_cohens_d": round(self._cohens_d(group1, group2), 6),
            "error": None
        }

    def _paired_t_test(self, data: Dict) -> dict:
        """Paired samples t-test."""
        before = np.array(data.get('before', []), dtype=float)
        after = np.array(data.get('after', []), dtype=float)
        alpha = data.get('alpha', 0.05)

        if len(before) != len(after):
            return {"error": "Before and after samples must have same length"}

        t_stat, p_value = stats.ttest_rel(before, after)

        return {
            "test": "Paired samples t-test",
            "t_statistic": round(float(t_stat), 6),
            "p_value": round(float(p_value), 6),
            "alpha": alpha,
            "significant": p_value < alpha,
            "interpretation": f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis at α={alpha}",
            "mean_difference": round(float(np.mean(after - before)), 6),
            "error": None
        }

    def _one_sample_t_test(self, data: Dict) -> dict:
        """One-sample t-test."""
        sample = np.array(data.get('sample', []), dtype=float)
        population_mean = data.get('population_mean', 0)
        alpha = data.get('alpha', 0.05)

        if len(sample) < 2:
            return {"error": "Sample needs at least 2 data points"}

        t_stat, p_value = stats.ttest_1samp(sample, population_mean)

        return {
            "test": "One-sample t-test",
            "t_statistic": round(float(t_stat), 6),
            "p_value": round(float(p_value), 6),
            "alpha": alpha,
            "significant": p_value < alpha,
            "sample_mean": round(float(np.mean(sample)), 6),
            "hypothesized_mean": population_mean,
            "interpretation": f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis at α={alpha}",
            "error": None
        }

    def _chi_square_test(self, data: Dict) -> dict:
        """Chi-square test for independence."""
        observed = np.array(data.get('observed', []), dtype=float)
        expected = data.get('expected', None)
        alpha = data.get('alpha', 0.05)

        if expected:
            expected = np.array(expected, dtype=float)
            chi2, p_value = stats.chisquare(observed, expected)
        else:
            # If 2D array, use chi-square test for independence
            if observed.ndim == 2:
                chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            else:
                chi2, p_value = stats.chisquare(observed)

        return {
            "test": "Chi-square test",
            "chi_square": round(float(chi2), 6),
            "p_value": round(float(p_value), 6),
            "alpha": alpha,
            "significant": p_value < alpha,
            "interpretation": f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis at α={alpha}",
            "error": None
        }

    def _anova_test(self, data: Dict) -> dict:
        """One-way ANOVA."""
        groups = data.get('groups', [])
        alpha = data.get('alpha', 0.05)

        if len(groups) < 2:
            return {"error": "ANOVA requires at least 2 groups"}

        # Convert to numpy arrays
        arrays = [np.array(g, dtype=float) for g in groups]

        f_stat, p_value = stats.f_oneway(*arrays)

        return {
            "test": "One-way ANOVA",
            "f_statistic": round(float(f_stat), 6),
            "p_value": round(float(p_value), 6),
            "alpha": alpha,
            "significant": p_value < alpha,
            "num_groups": len(groups),
            "interpretation": f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis at α={alpha}",
            "group_means": [round(float(np.mean(g)), 6) for g in arrays],
            "error": None
        }

    def _z_test(self, data: Dict) -> dict:
        """Z-test for population mean."""
        sample = np.array(data.get('sample', []), dtype=float)
        population_mean = data.get('population_mean', 0)
        population_std = data.get('population_std')
        alpha = data.get('alpha', 0.05)

        if population_std is None:
            return {"error": "Population standard deviation required for z-test"}

        sample_mean = np.mean(sample)
        n = len(sample)
        z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(n))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            "test": "Z-test",
            "z_statistic": round(float(z_stat), 6),
            "p_value": round(float(p_value), 6),
            "alpha": alpha,
            "significant": p_value < alpha,
            "sample_mean": round(float(sample_mean), 6),
            "interpretation": f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis at α={alpha}",
            "error": None
        }

    def _mann_whitney_test(self, data: Dict) -> dict:
        """Mann-Whitney U test (non-parametric)."""
        group1 = np.array(data.get('group1', []), dtype=float)
        group2 = np.array(data.get('group2', []), dtype=float)
        alpha = data.get('alpha', 0.05)

        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        return {
            "test": "Mann-Whitney U test",
            "u_statistic": round(float(u_stat), 6),
            "p_value": round(float(p_value), 6),
            "alpha": alpha,
            "significant": p_value < alpha,
            "interpretation": f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis at α={alpha}",
            "error": None
        }

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
