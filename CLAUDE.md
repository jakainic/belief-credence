# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for extracting and comparing credence signals from frontier LLMs. We implement three different methods to estimate P(True) for claims and compare their outputs:

1. **Direct Prompting**: Baseline method asking model for credence 0-1
2. **Logit Gap**: Analyzes P(True)/P(False) token probabilities
3. **CCS (Contrast Consistent Search)**: Discovers truth direction from statement/negation pairs

Additionally, we provide uncertainty estimation (hallucination probes) for validating credence estimates.

## Development Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package and dependencies
pip install -U pip
pip install -e . -r requirements.txt -r requirements-dev.txt

# Set up HuggingFace token (required for Llama models)
cp .env.example .env
# Edit .env and add your HF_TOKEN from https://huggingface.co/settings/tokens
```

## Common Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py

# Run specific test function
pytest tests/test_core.py::test_claim_creation
```

### Code Quality
```bash
# Run linter
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Type checking (strict mode)
mypy belief_credence
```

### Full CI Check
```bash
# Run the same checks as CI
ruff check . && mypy belief_credence && pytest -q
```

### Running Examples
```bash
# Compare all methods on sample claims
python examples/compare_methods.py

# Evaluate epistemological properties
python examples/evaluate_epistemology.py

# Search for best CCS layer
python examples/ccs_layer_search.py

# Comprehensive evaluation with datasets
python examples/evaluate_with_datasets.py
```

## Architecture

### Core Components

- **`core.py`**: Base classes and data structures
  - `Claim`: Represents a statement with optional negation
  - `CredenceEstimate`: Standardized P(True) output with metadata
  - `CredenceMethod`: Abstract base class for all methods

- **`model_utils.py`**: Model loading and activation extraction
  - `ModelWrapper`: Wraps HuggingFace models with utilities for:
    - Text generation
    - Logit extraction for specific tokens
    - Hidden state extraction from any layer
    - Contrast pair activation extraction

- **Credence Method Implementations** (3 methods):
  - `prompting.py`: Direct prompting - asks model for credence, parses numeric response
  - `logit_gap.py`: Extracts P("True")/P("False") token probabilities and normalizes
  - `ccs.py`: Trains linear probe on contrast pairs with consistency loss
    - Includes `search_best_layer()` utility for hyperparameter tuning
    - Finds optimal layer by comparing consistency scores across layers

- **`uncertainty.py`**: Uncertainty/confidence estimation (not a direct credence method)
  - `HallucinationProbe`: Linear probe on hidden states for uncertainty estimation
  - `UncertaintyEstimate`: Returns uncertainty_score and confidence_score
  - `check_credence_uncertainty_alignment()`: Validates credence vs uncertainty

- **`comparison.py`**: Utilities for running and comparing multiple methods
  - `compare_methods()`: Run all methods on one claim
  - `compare_on_dataset()`: Batch evaluation
  - `MethodComparison`: Results with statistics (mean, std, range)

- **`epistemology.py`**: Evaluation of Bayesian epistemological properties
  - `check_consistency()`: P(T) + P(F) ≈ 1 for contrast pairs
  - `check_coherence()`: Similar credences for paraphrases
  - `check_bayesian_conditioning()`: P(A|B) ≈ P(A∧B)/P(B)
  - `check_action_correlation()`: Internal credence vs action probability
  - `evaluate_epistemology()`: Comprehensive evaluation with report

- **`datasets.py`**: Curated contrastive claim pairs
  - Six belief categories: well-established facts, contested facts, certain/uncertain predictions, normative judgments, metaphysical beliefs
  - Multiple phrasings per claim for coherence testing
  - `ClaimSet`: Container for proposition with paraphrases
  - `get_dataset()`, `get_all_claims()`: Dataset loaders

### Design Principles

All methods implement the same interface:
```python
def estimate(self, claim: Claim) -> CredenceEstimate
```

This ensures outputs are comparable across methods - every method returns a standardized `p_true` estimate between 0 and 1.

### Model Integration

The library supports any HuggingFace model. Default is Llama-2-8b-hf, but you can use:
- `meta-llama/Llama-2-70b-hf` for larger model
- `meta-llama/Meta-Llama-3-8B` for Llama 3
- Any other CausalLM model on HuggingFace

All methods can share the same `ModelWrapper` instance to avoid loading multiple times:
```python
model = ModelWrapper("meta-llama/Llama-2-8b-hf", load_in_8bit=True)
method1 = DirectPrompting(model=model)
method2 = LogitGap(model=model)
```

## Implementation Status

### Completed
- ✓ Core architecture with `Claim`, `CredenceEstimate`, `CredenceMethod`
- ✓ Model loading and activation extraction infrastructure
- ✓ Three credence estimation methods (Direct Prompting, Logit Gap, CCS)
- ✓ CCS layer search utility for hyperparameter tuning
- ✓ Uncertainty estimation via hallucination probes (separate from credence methods)
- ✓ Comparison utilities for evaluating multiple methods
- ✓ Epistemological property evaluation:
  - Logical consistency (P(T) + P(F) ≈ 1)
  - Coherence across paraphrases
  - Bayesian conditioning (P(A|B) ≈ P(A∧B)/P(B))
  - Action-belief correlation
- ✓ Curated datasets:
  - 18 claim sets across 6 belief categories
  - 4 phrasings per claim for coherence testing
  - Well-established facts, contested facts, predictions, normative/metaphysical
- ✓ Example scripts (method comparison, epistemology, datasets, CCS layer search)
- ✓ Comprehensive unit tests

### Next Steps
To improve the methods, consider:
1. **Hallucination Probe**: Integrate pretrained probes from `github.com/obalcells/hallucination_probes`
2. **Logit Gap**: Experiment with different prompt formats (e.g., "Is the following true or false:")
3. **All Methods**: Test on real datasets and tune hyperparameters
4. **Evaluation**: Run comprehensive benchmarks across all belief categories

## External Code References

The implementations draw on these existing projects:
- CCS: `github.com/collin-burns/discovering_latent_knowledge` and `github.com/EleutherAI/elk`
- Logit lens: `github.com/dannyallover/overthinking_the_truth`
- Hallucination probes: `github.com/obalcells/hallucination_probes` and `github.com/OATML/semantic-entropy-probes`

## Python Coding Conventions

**Important rules from agents.md:**
- **Never use try/except clauses** unless specifically requested
- **Never use emojis** in print statements or code
- **Keep comments sparse** - prefer docstrings over inline comments
- **Never use hasattr() or getattr()** - reference attributes directly instead

## Configuration

- Python version: 3.11+ (enforced via strict mypy)
- Build system: setuptools
- Linter: Ruff (100 char line length, E/F/I/UP rules)
- Type checker: mypy (strict mode, warn on unused ignores)
- Test runner: pytest (quiet mode by default)
