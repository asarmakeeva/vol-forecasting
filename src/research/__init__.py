"""
Research Module for Volatility Forecasting
==========================================

Tools for GARCH model research and analysis.
"""

from .garch_analysis import (
    VolatilityComparison,
    GARCHDiagnostics,
    stylized_facts_summary,
    format_research_table
)

__all__ = [
    'VolatilityComparison',
    'GARCHDiagnostics',
    'stylized_facts_summary',
    'format_research_table'
]
