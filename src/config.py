"""
Central configuration for LLM Emotional Individuality Analysis.

This module contains all constants, paths, and settings used throughout the analysis.
Mathematical definitions align with the paper's theoretical framework.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
# Public-release default dataset (numeric-only).
# The original full dataset (with LLM-generated free-text justifications) is not redistributed here.
DATA_PATH = PROJECT_ROOT / "data_public.csv"
RESULTS_PATH = PROJECT_ROOT / "results"
CSV_OUTPUT = RESULTS_PATH / "csv"
JSON_OUTPUT = RESULTS_PATH / "json"
FIGURES_OUTPUT = RESULTS_PATH / "figures"
TABLES_OUTPUT = RESULTS_PATH / "tables"

# Ensure output directories exist
for path in [CSV_OUTPUT, JSON_OUTPUT, FIGURES_OUTPUT, TABLES_OUTPUT]:
    path.mkdir(parents=True, exist_ok=True)

# Emotion dimensions mapping
EMOTIONS = {
    "Q1": "Interest",
    "Q2": "Surprise",
    "Q3": "Sadness",
    "Q4": "Anger"
}

EMOTION_COLUMNS = ["Q1value", "Q2value", "Q3value", "Q4value"]
REASON_COLUMNS = ["Q1reason", "Q2reason", "Q3reason", "Q4reason"]

# Persona definitions with theoretical temperature assignments
PERSONAS = {
    "p1": {
        "name": "Student",
        "temperature": 0.7,
        "description": "First-year university student - intuitive, emotionally engaged"
    },
    "p2": {
        "name": "Researcher",
        "temperature": 0.4,
        "description": "Literary researcher - analytical, objective"
    },
    "p3": {
        "name": "Poet",
        "temperature": 0.9,
        "description": "Emotional poet - highly sensitive, creative"
    },
    "p4": {
        "name": "Robot",
        "temperature": 0.1,
        "description": "Emotionless robot - logical, consistent"
    }
}

# Text definitions
TEXTS = {
    "t1": {
        "title": "Kaichu-dokei (The Pocket Watch)",
        "author": "Yumeno Kyusaku",
        "year": 1923,
        "genre": "Allegorical",
        "description": "Allegorical dialogue between a pocket watch and mouse"
    },
    "t2": {
        "title": "Okane to Pisutoru (Money and Pistol)",
        "author": "Yumeno Kyusaku",
        "year": 1923,
        "genre": "Narrative",
        "description": "Narrative with socioeconomic themes"
    },
    "t3": {
        "title": "Boroboro na Dachou (The Ragged Ostrich)",
        "author": "Takamura Kotaro",
        "year": 1958,
        "genre": "Poetic",
        "description": "Poetic meditation on suffering"
    }
}

# Vendor categorization (7 major vendors)
VENDORS = [
    "OpenAI",
    "Anthropic",
    "Google",
    "xAI",
    "DeepSeek",
    "Meta",
    "Alibaba"
]

# Vendor color scheme for visualizations
VENDOR_COLORS = {
    "OpenAI": "#10a37f",
    "Anthropic": "#D97757",
    "Google": "#4285f4",
    "xAI": "#000000",
    "DeepSeek": "#7C4DFF",
    "Meta": "#0668E1",
    "Alibaba": "#FF6A00"
}

# Fuzzy membership function parameters
FUZZY_PARAMS = {
    "Low": {"start": 0, "peak": 0, "end": 30},
    "Medium": {"start": 10, "peak_start": 30, "peak_end": 50, "end": 70},
    "High": {"start": 50, "peak": 100, "end": 100}
}

# Temperature settings for analysis (used for documentation and defaults).
# The analysis code primarily uses the temperatures present in the input CSV.
TEMPERATURE_LEVELS = [0.1, 0.4, 0.7, 0.9]

# Reliability-Controllability matrix thresholds
RELIABILITY_THRESHOLD = 0.95
CONTROLLABILITY_THRESHOLD = 0.5

# Statistical parameters
ALPHA = 0.05
RANDOM_SEED = 42

# t-SNE parameters
TSNE_PARAMS = {
    "n_components": 2,
    "perplexity": 30,
    "n_iter": 1000,
    "random_state": RANDOM_SEED
}

# Visualization parameters
FIGURE_DPI = 300
FIGURE_SIZE = (10, 8)
FIGURE_FORMAT = ["png", "pdf"]
FIGURE_FONT_FAMILY = "serif"
FIGURE_FONT_SERIF = [
    "TeX Gyre Pagella",
    "Palatino Linotype",
    "Book Antiqua",
    "Times New Roman",
    "Nimbus Roman",
    "DejaVu Serif",
]
FIGURE_BASE_FONT_SIZE = 12
FIGURE_AXIS_LABEL_SIZE = FIGURE_BASE_FONT_SIZE
FIGURE_TICK_LABEL_SIZE = FIGURE_BASE_FONT_SIZE
FIGURE_LEGEND_FONT_SIZE = FIGURE_BASE_FONT_SIZE
FIGURE_LEGEND_TITLE_SIZE = FIGURE_BASE_FONT_SIZE
FIGURE_TITLE_SIZE = FIGURE_BASE_FONT_SIZE
FIGURE_ANNOTATION_SIZE = FIGURE_BASE_FONT_SIZE
FIGURE_PDF_FONT_TYPE = 42
FIGURE_PS_FONT_TYPE = 42
FIGURE_MATHTEXT_FONTSET = "stix"

# Expected data dimensions for the current dataset.
# - EXPECTED_DATA_POINTS is the designed maximum (texts × personas × trials × models)
# - EXPECTED_VALID_SCORE_RATE refers to valid score rows / collected rows
EXPECTED_DATA_POINTS = 4320
EXPECTED_VALID_SCORE_RATE = 0.962
EXPECTED_VALID_POINTS = 4067

# Paper validation values
PAPER_VALUES = {
    # Data collection summary (for reproducibility checks)
    "designed_trials": EXPECTED_DATA_POINTS,
    "collected_trials": 4227,
    "submission_coverage": 4227 / EXPECTED_DATA_POINTS,
    "valid_score_rate": 4067 / 4227,
    "valid_data_points": 4067,

    "temperature_corr_range": (-0.982, 0.967),
    "fuzzy_entropy_range": (0.404, 0.661),
    "persona_temp_synergy": {"r": 0.614, "p": 0.386},
    "consistency_range": (0.548, 0.780),
}

def get_persona_temperature(persona_id: str) -> float:
    """Get theoretical temperature assignment for a persona."""
    return PERSONAS[persona_id]["temperature"]

def get_emotion_name(emotion_code: str) -> str:
    """Get emotion name from code."""
    return EMOTIONS[emotion_code]
