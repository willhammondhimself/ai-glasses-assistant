"""Function graphing module for physics HUD."""

import io
import re
from typing import Tuple, Optional, List

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import sympy
from sympy import symbols, lambdify, sympify
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)


def plot_function(
    expr_str: str,
    x_range: Tuple[float, float] = (-10, 10),
    y_range: Optional[Tuple[float, float]] = None,
    num_points: int = 500,
    title: Optional[str] = None,
    style: str = "ar_green"
) -> bytes:
    """
    Plot a mathematical function and return as PNG bytes.

    Args:
        expr_str: Mathematical expression to plot (e.g., "x^2", "sin(x)")
        x_range: Tuple of (x_min, x_max)
        y_range: Optional tuple of (y_min, y_max), auto-calculated if None
        num_points: Number of points to plot
        title: Optional title for the plot
        style: Color scheme - "ar_green" (default), "ar_cyan", "classic"

    Returns:
        PNG image as bytes
    """
    # Parse expression
    x = symbols('x')
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    # Clean expression
    expr_str = _clean_expression(expr_str)

    # Parse to SymPy
    local_dict = {
        'x': x,
        'pi': sympy.pi,
        'e': sympy.E,
        'sin': sympy.sin,
        'cos': sympy.cos,
        'tan': sympy.tan,
        'log': sympy.log,
        'ln': sympy.ln,
        'exp': sympy.exp,
        'sqrt': sympy.sqrt,
        'abs': sympy.Abs,
    }

    try:
        expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Could not parse expression '{expr_str}': {e}")

    # Convert to numpy function
    f = lambdify(x, expr, modules=['numpy'])

    # Generate x values
    x_vals = np.linspace(x_range[0], x_range[1], num_points)

    # Calculate y values with error handling
    try:
        y_vals = f(x_vals)
    except Exception as e:
        raise ValueError(f"Could not evaluate expression: {e}")

    # Handle complex results (take real part)
    if np.iscomplexobj(y_vals):
        y_vals = np.real(y_vals)

    # Auto-calculate y_range if not provided
    if y_range is None:
        # Remove infinities and NaN for range calculation
        valid_y = y_vals[np.isfinite(y_vals)]
        if len(valid_y) > 0:
            y_min, y_max = np.min(valid_y), np.max(valid_y)
            y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
            y_range = (y_min - y_margin, y_max + y_margin)
        else:
            y_range = (-10, 10)

    # Create figure with style
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Style configuration
    styles = {
        "ar_green": {
            "line_color": "#00FF00",
            "grid_color": "#1a3a1a",
            "axis_color": "#00FF00",
            "text_color": "#00FF00",
        },
        "ar_cyan": {
            "line_color": "#00FFFF",
            "grid_color": "#1a3a3a",
            "axis_color": "#00FFFF",
            "text_color": "#00FFFF",
        },
        "classic": {
            "line_color": "#1f77b4",
            "grid_color": "#333333",
            "axis_color": "#ffffff",
            "text_color": "#ffffff",
        },
    }

    s = styles.get(style, styles["ar_cyan"])

    # Plot the function
    ax.plot(x_vals, y_vals, color=s["line_color"], linewidth=2, label=f'f(x) = {expr_str}')

    # Configure axes
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Grid
    ax.grid(True, color=s["grid_color"], linestyle='-', linewidth=0.5, alpha=0.7)

    # Axis lines through origin
    ax.axhline(y=0, color=s["axis_color"], linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color=s["axis_color"], linewidth=0.8, alpha=0.5)

    # Labels and title
    ax.set_xlabel('x', color=s["text_color"], fontsize=12)
    ax.set_ylabel('f(x)', color=s["text_color"], fontsize=12)

    if title:
        ax.set_title(title, color=s["text_color"], fontsize=14)
    else:
        ax.set_title(f'f(x) = {expr_str}', color=s["text_color"], fontsize=14)

    # Tick colors
    ax.tick_params(colors=s["text_color"])
    for spine in ax.spines.values():
        spine.set_color(s["axis_color"])
        spine.set_alpha(0.3)

    # Legend
    ax.legend(loc='best', facecolor='#1a1a1a', edgecolor=s["axis_color"],
              labelcolor=s["text_color"])

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#0a0a0a', edgecolor='none')
    buf.seek(0)

    plt.close(fig)

    return buf.read()


def plot_multiple_functions(
    expressions: List[str],
    x_range: Tuple[float, float] = (-10, 10),
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    style: str = "ar_cyan"
) -> bytes:
    """
    Plot multiple functions on the same graph.

    Args:
        expressions: List of mathematical expressions
        x_range: Tuple of (x_min, x_max)
        labels: Optional labels for each function
        title: Optional title for the plot
        style: Color scheme

    Returns:
        PNG image as bytes
    """
    x = symbols('x')
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    local_dict = {
        'x': x,
        'pi': sympy.pi,
        'e': sympy.E,
        'sin': sympy.sin,
        'cos': sympy.cos,
        'tan': sympy.tan,
        'log': sympy.log,
        'ln': sympy.ln,
        'exp': sympy.exp,
        'sqrt': sympy.sqrt,
        'abs': sympy.Abs,
    }

    # Colors for multiple functions
    colors = ['#00FFFF', '#00FF00', '#FF00FF', '#FFFF00', '#FF4040', '#40FF40']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    x_vals = np.linspace(x_range[0], x_range[1], 500)
    y_min, y_max = float('inf'), float('-inf')

    for i, expr_str in enumerate(expressions):
        expr_str = _clean_expression(expr_str)
        try:
            expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
            f = lambdify(x, expr, modules=['numpy'])
            y_vals = f(x_vals)

            if np.iscomplexobj(y_vals):
                y_vals = np.real(y_vals)

            # Track y range
            valid_y = y_vals[np.isfinite(y_vals)]
            if len(valid_y) > 0:
                y_min = min(y_min, np.min(valid_y))
                y_max = max(y_max, np.max(valid_y))

            label = labels[i] if labels and i < len(labels) else f'f{i+1}(x) = {expr_str}'
            color = colors[i % len(colors)]
            ax.plot(x_vals, y_vals, color=color, linewidth=2, label=label)

        except Exception as e:
            print(f"Could not plot '{expr_str}': {e}")
            continue

    # Configure axes
    ax.set_xlim(x_range)
    if y_min != float('inf') and y_max != float('-inf'):
        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Grid and styling
    ax.grid(True, color='#1a3a3a', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color='#00FFFF', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='#00FFFF', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('x', color='#00FFFF', fontsize=12)
    ax.set_ylabel('f(x)', color='#00FFFF', fontsize=12)

    if title:
        ax.set_title(title, color='#00FFFF', fontsize=14)

    ax.tick_params(colors='#00FFFF')
    for spine in ax.spines.values():
        spine.set_color('#00FFFF')
        spine.set_alpha(0.3)

    ax.legend(loc='best', facecolor='#1a1a1a', edgecolor='#00FFFF',
              labelcolor='#00FFFF')

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#0a0a0a', edgecolor='none')
    buf.seek(0)

    plt.close(fig)

    return buf.read()


def _clean_expression(expr: str) -> str:
    """Clean and normalize expression for parsing."""
    result = expr.strip()

    # LaTeX conversions
    conversions = [
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))'),
        (r'\\sqrt\{([^}]+)\}', r'sqrt(\1)'),
        (r'\^(\d)', r'**\1'),
        (r'\^\{([^}]+)\}', r'**(\1)'),
        (r'\\cdot', '*'),
        (r'\\times', '*'),
        (r'\\div', '/'),
        (r'\\pi', 'pi'),
        (r'\\sin', 'sin'),
        (r'\\cos', 'cos'),
        (r'\\tan', 'tan'),
        (r'\\log', 'log'),
        (r'\\ln', 'ln'),
        (r'\\exp', 'exp'),
        (r'\{', '('),
        (r'\}', ')'),
    ]

    for pattern, replacement in conversions:
        result = re.sub(pattern, replacement, result)

    # Handle implicit multiplication (2x -> 2*x)
    result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)

    return result


# Convenience function
def quick_plot(expr: str) -> bytes:
    """Quick plot with default settings."""
    return plot_function(expr)
