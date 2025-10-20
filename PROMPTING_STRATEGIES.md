# Prompting Strategies by Method

Each credence estimation method uses a different approach to query the model. Understanding these differences is crucial for interpreting results.

---

## 1. DirectPrompting

**Strategy:** Ask the model directly for a numerical probability.

**Prompt Format:**
```
Q: What is the probability that the following claim is true? Answer with only a number between 0 and 1.

Claim: "{statement}"

A:
```

**Example:**
```
Q: What is the probability that the following claim is true? Answer with only a number between 0 and 1.

Claim: "The Earth orbits around the Sun."

A:
```

**Expected Response:** `0.95` or `0.99` (model generates a number)

**How It Works:**
- Generates text continuation (max 10 tokens)
- Parses the first valid decimal number between 0 and 1
- Requires instruction-tuned models (e.g., `-chat-hf`)
- Most interpretable but least robust

**Failure Modes:**
- Base models may generate conversational text instead of numbers
- May default to anchoring values (0.5, 1.0)
- Sensitive to prompt wording

---

## 2. LogitGap

**Strategy:** Extract token probabilities for "True" and "False" tokens.

**Prompt Format:**
```
True or False: {statement}

Answer:
```

**Example:**
```
True or False: The Earth orbits around the Sun.

Answer:
```

**Expected Next Token:** Model assigns probabilities to "True" and "False" tokens

**How It Works:**
- Does NOT generate text
- Extracts logit probabilities: P("True") and P("False")
- Normalizes: `P(claim) = P("True") / (P("True") + P("False"))`
- Works with both base and chat models
- Based on Kadavath et al. (2022)

**Advantages:**
- Doesn't require text generation
- Works on base models
- More robust than DirectPrompting
- Directly measures model's internal confidence

**Limitations:**
- Only uses next-token probabilities (shallow signal)
- Assumes "True"/"False" tokens capture belief well

---

## 3. CCS (Contrast Consistent Search)

**Strategy:** Train a probe on internal activations to extract latent beliefs.

### 3a. Training Phase (uses prompts internally)

CCS needs to determine "what the model believes" to orient the probe. It uses one of two sub-methods:

#### Method 1: LogitGap (default)
Uses the same LogitGap prompt internally:
```
True or False: {statement}

Answer:
```

Determines belief by checking if P("True") > P("False")

#### Method 2: Self-Label
Asks model to choose between statement and negation:
```
Which statement is true?
A: {positive_statement}
B: {negative_statement}

Answer:
```

Determines belief by checking if P("A") > P("B")

### 3b. Evaluation Phase (Raw statement, no special prompt)

Once trained, CCS evaluation uses the raw statement without special formatting:

**How It Works:**
1. Feed raw statement through model: `"The Earth orbits around the Sun."`
2. Model processes it and generates hidden states at each layer
3. Extract hidden states from specified layer (e.g., layer -1)
4. Apply trained linear probe to those hidden states
5. Probe outputs probability directly

**Key Difference from Other Methods:**
- DirectPrompting: Adds "Q: What is the probability...?" wrapper
- LogitGap: Adds "True or False:" prefix
- CCS: Uses bare statement - just `"The Earth orbits around the Sun."`

**Advantages:**
- Accesses deeper representations than logits
- Minimal prompt engineering (just raw claim text)
- Can extract beliefs the model doesn't verbalize well
- Based on Burns et al. (2022)

**Limitations:**
- Requires training data with negations
- Probe orientation depends on training method (LogitGap or self-label)
- More complex than other methods
- May not work if representations are unclear

---

## Comparison Summary

| Method | Prompt at Test Time? | Generates Text? | Works on Base Models? | Complexity |
|--------|---------------------|----------------|----------------------|------------|
| **DirectPrompting** | Yes (Q&A format) | Yes | No (requires chat) | Low |
| **LogitGap** | Yes (True/False) | No | Yes | Low |
| **CCS** | Minimal (raw text) | No | Yes | High |

**Signals Used:**
- **DirectPrompting**: Generated text → parsed number
- **LogitGap**: Next-token logits (P("True"), P("False"))
- **CCS**: Internal hidden states from raw statement (layer activations)

---

## Why Different Prompts Matter

### For DirectPrompting
The prompt heavily influences output quality:
- Too verbose: Model may generate explanations instead of numbers
- Too terse: Model may not understand the task
- Suggestive examples: Model may anchor on them (0.5 problem)

**Current prompt** uses Q&A format, which chat models handle well.

### For LogitGap
The "True or False" format is optimal because:
- Simple binary choice
- "True" and "False" are single tokens in most tokenizers
- Minimal prompt engineering needed
- Works consistently across models

### For CCS
During training, the orientation prompt matters:
- **LogitGap orientation** (default): More reliable, uses same signal
- **Self-label orientation**: May work better for ambiguous claims

At test time, minimal formatting - just feeds the raw statement to get activations!

---

## Practical Implications

### When Results Disagree

If methods give different probabilities for the same claim, consider:

1. **DirectPrompting vs LogitGap disagreement**
   - DirectPrompting may be misinterpreting the prompt
   - LogitGap is more robust, trust it more

2. **LogitGap vs CCS disagreement**
   - LogitGap uses surface-level signal (next token)
   - CCS uses deeper representations
   - If CCS fails on well-established facts, its training may have failed

3. **All three disagree**
   - The model may genuinely be uncertain
   - Or: the claim is ambiguous/poorly formed

### Prompt Engineering

**Should you tune the prompts?**

- **DirectPrompting**: YES - very sensitive to wording
- **LogitGap**: MAYBE - current prompt is standard and works well
- **CCS**: MINIMAL - uses raw text at test time, but orientation method during training can be changed

---

## Implementation Notes

**Where prompts are defined:**
- DirectPrompting: `belief_credence/prompting.py:_build_prompt()`
- LogitGap: `belief_credence/logit_gap.py:_format_prompt()`
- CCS training:
  - LogitGap method: `belief_credence/ccs.py:_get_model_belief_logit_gap()`
  - Self-label method: `belief_credence/ccs.py:_get_model_belief_self_label()`

**To customize prompts:**
Edit the corresponding `_build_prompt()` or `_format_prompt()` method in each file.

---

## References

- **LogitGap**: Kadavath et al. (2022) "Language Models (Mostly) Know What They Know" - https://arxiv.org/abs/2207.05221
- **CCS**: Burns et al. (2022) "Discovering Latent Knowledge in Language Models Without Supervision" - https://arxiv.org/abs/2212.03827
