# Mathematical Framework for Characterizing Emotional Individuality in Large Language Models

This repository contains the analysis code and a **numeric-only** dataset (`data_public.csv`) supporting our MDPI submission:

**"Mathematical Framework for Characterizing Emotional Individuality in Large Language Models: Temperature Control, Fuzzy Entropy, and Persona-Based Diversity Analysis"**

**Paper link / DOI**: TBD (will be added after publication).

## What is included

- **Reproducible analysis code** (`src/`, `main.py`, `verify_results.py`)
- **Numeric-only dataset** (`data_public.csv`): scores + metadata only

### What is NOT included

- Manuscript sources (e.g., LaTeX / PDF) are not distributed here.
- The full dataset that contains **LLM-generated free-text justifications** (e.g., `Q1reason`--`Q4reason`) is not redistributed in this repository. Some providers' terms may restrict redistribution of generated text; the analyses reported in the paper do not rely on these free-text fields.

## Repository structure

```
llm-emotional-individuality/
├── data_public.csv                 # Numeric-only dataset (scores + metadata)
├── src/                            # Analysis modules
├── main.py                         # Main pipeline
├── verify_results.py               # Checks key aggregates vs paper baselines
├── requirements.txt                # Python dependencies
├── LICENSE                         # Code license (MIT)
└── LICENSE-DATA.md                 # Data license (CC BY 4.0)
```

## Installation

- Python 3.10+

```bash
git clone https://github.com/nshrhm/llm-emotional-individuality.git
cd llm-emotional-individuality
pip install -r requirements.txt
```

## Reproduce the results

```bash
python main.py
python verify_results.py
```

By default, the analysis reads `data_public.csv`. Outputs are generated under `results/` (not committed).

## Data

`data_public.csv` contains:

- Metadata: `timestamp`, `text` (IDs `t1`--`t3`), `developer`, `model`, `persona`, `temperature`, `trial`
- Scores: `Q1value` (Interest), `Q2value` (Surprise), `Q3value` (Sadness), `Q4value` (Anger)

## Licensing

- Code: MIT License (see `LICENSE`)
- Data: CC BY 4.0 (see `LICENSE-DATA.md`)

