# belief-credence

Research library for extracting and comparing credence signals from frontier LLMs.

## Overview

This library implements three different methods for estimating P(True) for claims, allowing you to compare how different approaches detect credence/belief signals in language models:

1. **Direct Prompting**: Ask the model directly for a credence value (0-1)
2. **Logit Gap**: Extract P("True")/P("False") token probabilities ([Kadavath et al. 2022](https://arxiv.org/abs/2207.05221))
3. **CCS (Contrast Consistent Search)**: Find truth direction from statement/negation pairs ([Burns et al. 2022](https://arxiv.org/abs/2212.03827))

All methods output a standardized `CredenceEstimate` with `p_true` between 0 and 1, making results directly comparable.

Additionally, the library includes **uncertainty estimation** via hallucination probes to assess model confidence (not a direct credence method, but useful for validation).

## Installation

```bash
git clone https://github.com/yourusername/belief-credence
cd belief-credence
pip install -e . -r requirements.txt
```

### HuggingFace Authentication

For gated models like Llama 2/3, you need a HuggingFace token:

1. Get your token from https://huggingface.co/settings/tokens
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Add your token to `.env`:
   ```
   HF_TOKEN=your_token_here
   ```

The token will be automatically loaded when you use `ModelWrapper`.

## Quick Start

```python
from belief_credence import Claim, DirectPrompting, LogitGap, compare_methods
from belief_credence.model_utils import ModelWrapper

# Load model once, share across methods
model = ModelWrapper("meta-llama/Llama-2-8b-hf", load_in_8bit=True)

# Create claim with negation (needed for CCS)
claim = Claim(
    statement="The Eiffel Tower is located in Paris.",
    negation="The Eiffel Tower is not located in Paris."
)

# Initialize methods
methods = [
    DirectPrompting(model=model),
    LogitGap(model=model),
]

# Compare all methods
comparison = compare_methods(claim, methods)

# View results
for method_name, estimate in comparison.estimates.items():
    print(f"{method_name}: P(True) = {estimate.p_true:.3f}")

print(f"\nMean: {comparison.mean_p_true():.3f}")
print(f"Std: {comparison.std_p_true():.3f}")
```

## Methods

### Direct Prompting
Simplest baseline - prompts model to output a credence value and parses the response.

```python
from belief_credence import DirectPrompting, Claim

method = DirectPrompting(model_name="meta-llama/Llama-2-8b-hf")
claim = Claim(statement="Python is a programming language.")
estimate = method.estimate(claim)
print(estimate.p_true)  # e.g., 0.95
```

### Logit Gap
Analyzes token probabilities for "True" and "False" tokens.

```python
from belief_credence import LogitGap, Claim

method = LogitGap(model_name="meta-llama/Llama-2-8b-hf")
claim = Claim(statement="The Earth is flat.")
estimate = method.estimate(claim)
print(estimate.p_true)  # e.g., 0.02
print(estimate.metadata["p_true_raw"])  # Raw P("True")
print(estimate.metadata["p_false_raw"])  # Raw P("False")
```

### CCS (Contrast Consistent Search)
Trains a linear probe to find truth direction from contrast pairs.

```python
from belief_credence import CCS, Claim

method = CCS(model_name="meta-llama/Llama-2-8b-hf", layer=-1)

# Train on multiple claims with negations
training_claims = [
    Claim("Paris is in France.", "Paris is not in France."),
    Claim("2+2=4", "2+2≠4"),
    # ... more pairs
]
method.train_probe(training_claims)

# Evaluate new claim
claim = Claim("The sky is blue.", "The sky is not blue.")
estimate = method.estimate(claim)
print(estimate.p_true)
print(estimate.metadata["consistency_score"])  # Lower is better
```

#### Layer Search for CCS
The layer hyperparameter can significantly impact CCS performance. Use `search_best_layer()` to find the optimal layer:

```python
from belief_credence import search_best_layer, get_dataset, BeliefType
from belief_credence.model_utils import ModelWrapper

# Load model
model = ModelWrapper("meta-llama/Llama-2-8b-hf", load_in_8bit=True)

# Get training data
claim_sets = get_dataset(BeliefType.WELL_ESTABLISHED_FACT)
training_claims = [cs.to_claims()[0] for cs in claim_sets]

# Search layers -1 through -5
result = search_best_layer(
    model=model,
    training_claims=training_claims,
    layers=[-1, -2, -3, -4, -5]
)

print(f"Best layer: {result.layer}")
print(f"Consistency score: {result.consistency_score:.3f}")

# Use the best layer
method = CCS(model=model, layer=result.layer)
method._probe = result.probe  # Use the trained probe
```

### Uncertainty Estimation (Hallucination Probes)
Estimates model uncertainty/confidence using hallucination detection probes.
**Note:** This provides uncertainty scores, not direct P(True) estimates.

```python
from belief_credence import HallucinationProbe, Claim

probe = HallucinationProbe(model_name="meta-llama/Llama-2-8b-hf", layer=-1)

claim = Claim(statement="Barack Obama was born in Hawaii.")
uncertainty = probe.estimate_uncertainty(claim)

print(f"Uncertainty: {uncertainty.uncertainty_score:.3f}")
print(f"Confidence: {uncertainty.confidence_score:.3f}")

# Check if a credence estimate aligns with uncertainty
from belief_credence import check_credence_uncertainty_alignment

p_true = 0.95  # from another method
is_aligned = check_credence_uncertainty_alignment(p_true, uncertainty)
print(f"Aligned: {is_aligned}")  # High p_true should have low uncertainty
```

## Development

```bash
# Install dev dependencies
pip install -e . -r requirements.txt -r requirements-dev.txt

# Run tests
pytest

# Type checking
mypy belief_credence

# Linting
ruff check .
```

## Architecture

All methods implement the `CredenceMethod` interface:
```python
class CredenceMethod(ABC):
    @abstractmethod
    def estimate(self, claim: Claim) -> CredenceEstimate:
        pass
```

This ensures consistent outputs with:
- `p_true`: Probability estimate (0-1)
- `method`: Method name
- `claim`: Original claim
- `raw_output`: Method-specific raw data
- `metadata`: Additional info for debugging

See `examples/compare_methods.py` for a complete usage example.

## Epistemological Property Evaluation

Beyond just extracting credences, this library can evaluate whether the estimates follow rational Bayesian principles:

### 1. Logical Consistency
Check if P(statement) + P(negation) ≈ 1:

```python
from belief_credence import check_consistency, Claim

claim = Claim(
    statement="The Earth orbits the Sun.",
    negation="The Earth does not orbit the Sun."
)

result = check_consistency(method, claim)
print(f"P(statement): {result.p_statement:.3f}")
print(f"P(negation): {result.p_negation:.3f}")
print(f"Sum: {result.sum_probability:.3f}")
print(f"Consistent? {result.is_consistent}")
print(f"Score: {result.consistency_score:.3f}")
```

### 2. Coherence Across Paraphrases
Check if semantically equivalent statements get similar credences:

```python
from belief_credence import check_coherence

claim = Claim(statement="Paris is the capital of France.")
paraphrases = [
    "France's capital is Paris.",
    "The capital city of France is Paris.",
]

result = check_coherence(method, claim, paraphrases)
print(f"Std deviation: {result.std_deviation:.3f}")
print(f"Coherent? {result.is_coherent}")
```

### 3. Bayesian Conditioning
Check if P(A|B) ≈ P(A ∧ B) / P(B):

```python
from belief_credence import check_bayesian_conditioning

result = check_bayesian_conditioning(
    method,
    proposition="the ground is wet",
    evidence="it is raining"
)

print(f"P(A|B) measured: {result.p_a_given_b:.3f}")
print(f"P(A|B) expected: {result.expected_p_a_given_b:.3f}")
print(f"Bayesian? {result.is_bayesian}")
```

### 4. Action-Belief Correlation
Check if internal credence correlates with action probabilities:

```python
from belief_credence import check_action_correlation

claim = Claim(statement="The Earth is round.")
action_prompt = "Would you bet $100 that: {claim.statement}"

result = check_action_correlation(method, claim, action_prompt)
print(f"Internal credence: {result.internal_credence:.3f}")
print(f"Action probability: {result.action_probability:.3f}")
print(f"Aligned? {result.is_aligned}")
```

### Comprehensive Evaluation
Run all checks at once:

```python
from belief_credence import evaluate_epistemology

report = evaluate_epistemology(
    method,
    consistency_claims=[...],
    coherence_tests=[...],
    bayesian_tests=[...],
    action_tests=[...]
)

print(f"Overall consistency: {report.overall_consistency_score:.3f}")
print(f"Overall coherence: {report.overall_coherence_score:.3f}")
print(f"Overall Bayesian: {report.overall_bayesian_score:.3f}")
print(f"Overall action: {report.overall_action_score:.3f}")
print(f"Overall epistemology: {report.overall_epistemology_score:.3f}")
```

See `examples/evaluate_epistemology.py` for a complete example.

## Curated Datasets

The library includes curated datasets of contrastive claim pairs across six belief categories, each with multiple phrasings to test consistency and coherence:

### Belief Categories

1. **Well-Established Facts** - Scientific and geographic facts with strong consensus
   - Example: "The Earth orbits around the Sun" (4 phrasings)

2. **Contested Facts** - Empirical claims with ongoing debate
   - Example: "Human activity is the primary cause of recent global warming" (4 phrasings)

3. **Certain Predictions** - High-confidence future events
   - Example: "The Sun will rise tomorrow morning" (4 phrasings)

4. **Uncertain Predictions** - Speculative future events
   - Example: "Artificial general intelligence will be developed by 2050" (4 phrasings)

5. **Normative Judgments** - Moral and political value claims
   - Example: "Lying is morally wrong" (4 phrasings)

6. **Metaphysical Beliefs** - Philosophical positions
   - Example: "Free will exists" (4 phrasings)

### Usage

```python
from belief_credence import BeliefType, get_dataset, get_all_claims

# Get claims for a specific category
well_established = get_dataset(BeliefType.WELL_ESTABLISHED_FACT)
for claim_set in well_established:
    print(claim_set.description)
    print(f"  Positive: {claim_set.positive_phrasings[0]}")
    print(f"  Negative: {claim_set.negative_phrasings[0]}")

# Get all claims across categories
all_claims = get_all_claims()
print(f"Total claims: {len(all_claims)}")

# Filter by category
metaphysical_claims = get_all_claims(BeliefType.METAPHYSICAL_BELIEF)
```

### Evaluation with Datasets

```python
from belief_credence import evaluate_epistemology, get_dataset, BeliefType

# Evaluate consistency across a belief category
claim_sets = get_dataset(BeliefType.WELL_ESTABLISHED_FACT)
consistency_claims = [cs.to_claims()[0] for cs in claim_sets]
coherence_tests = [(cs.to_claims()[0], cs.positive_phrasings[1:]) for cs in claim_sets]

report = evaluate_epistemology(
    method,
    consistency_claims=consistency_claims,
    coherence_tests=coherence_tests
)

print(f"Consistency: {report.overall_consistency_score:.3f}")
print(f"Coherence: {report.overall_coherence_score:.3f}")
```

See `examples/evaluate_with_datasets.py` for comprehensive examples.

## References

- **CCS**: [Burns et al. 2022 - Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827)
- **Logit Gap**: [Kadavath et al. 2022 - Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)
- **Hallucination Probes**: [hallucination-probes.com](https://www.hallucination-probes.com/)
