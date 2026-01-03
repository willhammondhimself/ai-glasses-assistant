"""
OptionsEngine: Options pricing and Greeks for quant finance interviews.

Pure Python implementation (educational - shows the math).

Features:
- Black-Scholes pricing for calls and puts
- All Greeks: delta, gamma, theta, vega, rho
- Implied volatility via Newton-Raphson
- Put-call parity verification
"""

import math
from typing import Optional


class OptionsEngine:
    """Options pricing engine with Black-Scholes model."""

    def __init__(self):
        pass

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """
        Standard normal cumulative distribution function.
        Uses approximation from Abramowitz and Stegun.
        """
        # Constants for approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)

        return 0.5 * (1.0 + sign * y)

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal probability density function."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def _d1(self, S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate d1 in Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return 0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    def _d2(self, S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate d2 in Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return 0
        return self._d1(S, K, r, sigma, T) - sigma * math.sqrt(T)

    def black_scholes(
        self,
        S: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: str = 'call'
    ) -> dict:
        """
        Calculate Black-Scholes option price.

        Args:
            S: Current stock price
            K: Strike price
            r: Risk-free interest rate (annualized, decimal)
            sigma: Volatility (annualized, decimal)
            T: Time to expiration (in years)
            option_type: 'call' or 'put'

        Returns:
            dict with price, d1, d2, and formula explanation
        """
        if S <= 0 or K <= 0:
            return {"error": "Stock price and strike must be positive"}
        if T < 0:
            return {"error": "Time to expiration cannot be negative"}
        if sigma < 0:
            return {"error": "Volatility cannot be negative"}

        option_type = option_type.lower()
        if option_type not in ['call', 'put']:
            return {"error": "option_type must be 'call' or 'put'"}

        # Handle edge case: at expiration
        if T == 0:
            if option_type == 'call':
                price = max(S - K, 0)
            else:
                price = max(K - S, 0)
            return {
                'price': round(price, 4),
                'd1': None,
                'd2': None,
                'formula': 'At expiration: intrinsic value only',
                'error': None
            }

        d1 = self._d1(S, K, r, sigma, T)
        d2 = self._d2(S, K, r, sigma, T)

        if option_type == 'call':
            price = S * self._norm_cdf(d1) - K * math.exp(-r * T) * self._norm_cdf(d2)
            formula = f"C = S·N(d1) - K·e^(-rT)·N(d2) = {S}·N({d1:.4f}) - {K}·e^(-{r}·{T})·N({d2:.4f})"
        else:
            price = K * math.exp(-r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)
            formula = f"P = K·e^(-rT)·N(-d2) - S·N(-d1) = {K}·e^(-{r}·{T})·N({-d2:.4f}) - {S}·N({-d1:.4f})"

        return {
            'price': round(price, 4),
            'option_type': option_type,
            'd1': round(d1, 6),
            'd2': round(d2, 6),
            'N_d1': round(self._norm_cdf(d1), 6),
            'N_d2': round(self._norm_cdf(d2), 6),
            'formula': formula,
            'inputs': {
                'S': S,
                'K': K,
                'r': r,
                'sigma': sigma,
                'T': T
            },
            'error': None
        }

    def greeks(
        self,
        S: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: str = 'call'
    ) -> dict:
        """
        Calculate all Greeks for an option.

        Returns:
            dict with delta, gamma, theta, vega, rho
        """
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return {"error": "Invalid inputs: all values must be positive"}

        d1 = self._d1(S, K, r, sigma, T)
        d2 = self._d2(S, K, r, sigma, T)

        # Common terms
        sqrt_T = math.sqrt(T)
        exp_rT = math.exp(-r * T)
        n_d1 = self._norm_pdf(d1)
        N_d1 = self._norm_cdf(d1)
        N_d2 = self._norm_cdf(d2)

        # Gamma (same for call and put)
        gamma = n_d1 / (S * sigma * sqrt_T)

        # Vega (same for call and put) - per 1% change in volatility
        vega = S * sqrt_T * n_d1 / 100

        if option_type.lower() == 'call':
            delta = N_d1
            theta = (-(S * n_d1 * sigma) / (2 * sqrt_T)
                     - r * K * exp_rT * N_d2) / 365  # Per day
            rho = K * T * exp_rT * N_d2 / 100  # Per 1% change in rate
        else:
            delta = N_d1 - 1
            theta = (-(S * n_d1 * sigma) / (2 * sqrt_T)
                     + r * K * exp_rT * self._norm_cdf(-d2)) / 365
            rho = -K * T * exp_rT * self._norm_cdf(-d2) / 100

        return {
            'delta': round(delta, 6),
            'gamma': round(gamma, 6),
            'theta': round(theta, 6),
            'vega': round(vega, 6),
            'rho': round(rho, 6),
            'option_type': option_type,
            'explanations': {
                'delta': f"Change in option price per $1 change in stock: {delta:.4f}",
                'gamma': f"Change in delta per $1 change in stock: {gamma:.6f}",
                'theta': f"Daily time decay: ${theta:.4f} per day",
                'vega': f"Change in price per 1% change in volatility: ${vega:.4f}",
                'rho': f"Change in price per 1% change in interest rate: ${rho:.4f}"
            },
            'error': None
        }

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        r: float,
        T: float,
        option_type: str = 'call',
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> dict:
        """
        Calculate implied volatility using Newton-Raphson method.

        Args:
            market_price: Observed option price in market
            S, K, r, T: Black-Scholes inputs
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance

        Returns:
            dict with implied volatility and convergence info
        """
        if market_price <= 0:
            return {"error": "Market price must be positive"}

        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = math.sqrt(2 * math.pi / T) * market_price / S

        for i in range(max_iterations):
            # Calculate price at current sigma
            bs_result = self.black_scholes(S, K, r, sigma, T, option_type)
            if bs_result.get('error'):
                return {"error": f"Black-Scholes error at iteration {i}"}

            price = bs_result['price']
            diff = price - market_price

            # Check convergence
            if abs(diff) < tolerance:
                return {
                    'implied_volatility': round(sigma, 6),
                    'implied_volatility_pct': f"{sigma * 100:.2f}%",
                    'iterations': i + 1,
                    'converged': True,
                    'price_error': round(diff, 8),
                    'method': 'Newton-Raphson',
                    'error': None
                }

            # Calculate vega for Newton-Raphson update
            d1 = self._d1(S, K, r, sigma, T)
            vega = S * math.sqrt(T) * self._norm_pdf(d1)

            if vega < 1e-10:
                return {"error": "Vega too small, cannot converge"}

            # Update sigma
            sigma = sigma - diff / vega

            # Keep sigma in reasonable bounds
            sigma = max(0.001, min(5.0, sigma))

        return {
            'implied_volatility': round(sigma, 6),
            'implied_volatility_pct': f"{sigma * 100:.2f}%",
            'iterations': max_iterations,
            'converged': False,
            'price_error': round(diff, 8),
            'method': 'Newton-Raphson',
            'warning': 'Did not converge within max iterations',
            'error': None
        }

    def parity_check(
        self,
        call_price: float,
        put_price: float,
        S: float,
        K: float,
        r: float,
        T: float
    ) -> dict:
        """
        Verify put-call parity: C - P = S - K*e^(-rT)

        Args:
            call_price: Market call price
            put_price: Market put price
            S, K, r, T: Option parameters

        Returns:
            dict with parity check results and arbitrage opportunity
        """
        # Theoretical relationship
        pv_strike = K * math.exp(-r * T)
        theoretical_diff = S - pv_strike

        # Actual difference
        actual_diff = call_price - put_price

        # Check for arbitrage
        parity_error = actual_diff - theoretical_diff

        # Determine if arbitrage exists (allowing for transaction costs)
        threshold = 0.05  # 5 cents tolerance

        if abs(parity_error) <= threshold:
            arbitrage = "No arbitrage (within tolerance)"
            strategy = None
        elif parity_error > threshold:
            arbitrage = "Call overpriced or Put underpriced"
            strategy = "Sell call, buy put, buy stock, borrow PV(K)"
        else:
            arbitrage = "Put overpriced or Call underpriced"
            strategy = "Buy call, sell put, sell stock, invest PV(K)"

        return {
            'parity_formula': 'C - P = S - K·e^(-rT)',
            'call_price': call_price,
            'put_price': put_price,
            'actual_diff': round(actual_diff, 4),
            'theoretical_diff': round(theoretical_diff, 4),
            'parity_error': round(parity_error, 4),
            'pv_strike': round(pv_strike, 4),
            'parity_holds': abs(parity_error) <= threshold,
            'arbitrage_opportunity': arbitrage,
            'strategy': strategy,
            'error': None
        }

    def option_chain_analysis(
        self,
        S: float,
        strikes: list,
        r: float,
        sigma: float,
        T: float
    ) -> dict:
        """
        Generate option chain with prices and Greeks for multiple strikes.

        Args:
            S: Current stock price
            strikes: List of strike prices
            r: Risk-free rate
            sigma: Volatility
            T: Time to expiration

        Returns:
            dict with call and put data for each strike
        """
        chain = []

        for K in strikes:
            call_bs = self.black_scholes(S, K, r, sigma, T, 'call')
            put_bs = self.black_scholes(S, K, r, sigma, T, 'put')
            call_greeks = self.greeks(S, K, r, sigma, T, 'call')
            put_greeks = self.greeks(S, K, r, sigma, T, 'put')

            chain.append({
                'strike': K,
                'moneyness': 'ITM' if S > K else ('ATM' if S == K else 'OTM'),
                'call': {
                    'price': call_bs.get('price'),
                    'delta': call_greeks.get('delta'),
                    'gamma': call_greeks.get('gamma'),
                    'theta': call_greeks.get('theta'),
                    'vega': call_greeks.get('vega'),
                },
                'put': {
                    'price': put_bs.get('price'),
                    'delta': put_greeks.get('delta'),
                    'gamma': put_greeks.get('gamma'),
                    'theta': put_greeks.get('theta'),
                    'vega': put_greeks.get('vega'),
                }
            })

        return {
            'spot_price': S,
            'volatility': sigma,
            'risk_free_rate': r,
            'time_to_expiry': T,
            'chain': chain,
            'error': None
        }

    def get_formulas(self) -> dict:
        """Return Black-Scholes formulas for reference."""
        return {
            'black_scholes_call': 'C = S·N(d₁) - K·e^(-rT)·N(d₂)',
            'black_scholes_put': 'P = K·e^(-rT)·N(-d₂) - S·N(-d₁)',
            'd1': 'd₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)',
            'd2': 'd₂ = d₁ - σ√T',
            'put_call_parity': 'C - P = S - K·e^(-rT)',
            'greeks': {
                'delta_call': 'Δ_call = N(d₁)',
                'delta_put': 'Δ_put = N(d₁) - 1',
                'gamma': 'Γ = N\'(d₁) / (S·σ·√T)',
                'theta_call': 'Θ_call = -S·N\'(d₁)·σ/(2√T) - r·K·e^(-rT)·N(d₂)',
                'vega': 'ν = S·√T·N\'(d₁)',
                'rho_call': 'ρ_call = K·T·e^(-rT)·N(d₂)'
            },
            'variables': {
                'S': 'Current stock price',
                'K': 'Strike price',
                'r': 'Risk-free interest rate (annualized)',
                'σ': 'Volatility (annualized)',
                'T': 'Time to expiration (years)',
                'N(x)': 'Standard normal CDF',
                'N\'(x)': 'Standard normal PDF'
            }
        }
