# Method Dependencies on Contrastive Pairs

This document explains which methods require negations and which can work independently.

## Methods That Work Independently

These methods can evaluate single statements without negations:

### 1. Direct Prompting
- **Requires negation**: ❌ No
- **How it works**: Prompts model to output a credence value 0-1 for the statement
- **Independence**: Fully independent - each statement evaluated separately
- **Usage**:
  ```python
  claim = Claim(statement="The Earth is round.")  # No negation needed
  estimate = DirectPrompting(model=model).estimate(claim)
  ```

### 2. Logit Gap
- **Requires negation**: ❌ No
- **How it works**: Extracts P("True") and P("False") token probabilities after prompting "True or False: {statement}"
- **Independence**: Fully independent - each statement evaluated separately
- **Usage**:
  ```python
  claim = Claim(statement="Water freezes at 0°C.")  # No negation needed
  estimate = LogitGap(model=model).estimate(claim)
  ```

## Methods That Require Contrastive Pairs (For Training Only)

### 3. CCS (Contrast Consistent Search)
- **Requires negation for training**: ✅ Yes
- **Requires negation for evaluation**: ❌ No
- **How it works**:
  - **Training**: Learns a probe from activation differences between statement/negation pairs. Uses direction method (logit_gap or self_label) to orient training data so probe learns "high = model believes true"
  - **Evaluation**: Simply applies trained probe to single statement activations. No negation needed!
- **Dependencies**:
  - Training requires multiple statement/negation pairs
  - Evaluation is fully independent - just needs the statement
  - Direction disambiguation only happens during training
- **Usage**:
  ```python
  # Training requires negations
  training_claims = [
      Claim("Paris is in France.", "Paris is not in France."),
      Claim("2+2=4", "2+2≠4"),
  ]
  ccs = CCS(model=model, direction_method="logit_gap")
  ccs.train_probe(training_claims)

  # Evaluation does NOT require negation!
  claim = Claim(statement="The sky is blue.")  # No negation needed
  estimate = ccs.estimate(claim)
  ```

## Batch Evaluation Considerations

When batch evaluating and saving outputs:

**For all methods (Direct Prompting, Logit Gap, and trained CCS)**:
- All can evaluate any list of statements independently
- No dependency between evaluations
- Results are fully cacheable
- No negations needed for evaluation

**For CCS training**:
- Must provide negations for training claims
- Direction method (logit_gap or self_label) only runs during training
- Once trained, CCS probe can be reused across any statements

## Recommendation for Efficient Batch Evaluation

For maximum efficiency when comparing all methods:

```python
from belief_credence import DirectPrompting, LogitGap, CCS, batch_evaluate_methods, get_dataset, BeliefType

# Step 1: Train CCS on claims with negations
training_claim_sets = get_dataset(BeliefType.WELL_ESTABLISHED_FACT)
training_claims = [cs.to_claims()[0] for cs in training_claim_sets]  # Has negations

ccs = CCS(model=model, direction_method="logit_gap")
ccs.train_probe(training_claims)  # Direction method used here

# Step 2: Evaluate on ANY statements (no negations needed!)
eval_statements = [
    Claim(statement="Earth orbits Sun."),
    Claim(statement="Water is wet."),
    Claim(statement="Python is a language."),
    # ... can be any statements, no negations required
]

# All methods work independently now
methods = [
    DirectPrompting(model=model),
    LogitGap(model=model),
    ccs,  # Already trained
]

# Save all results
results = batch_evaluate_methods(methods, eval_statements, output_dir="outputs/")
```

## Summary

**Key Insight**: After initial training, CCS becomes just as independent as Direct Prompting and Logit Gap. All three methods can evaluate arbitrary single statements without needing negations or contrastive pairs.
