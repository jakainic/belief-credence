# RunPod Evaluation Scripts

Scripts for running evaluations on RunPod and generating comparison visualizations.

## Quick Start

### 1. Setup on RunPod

```bash
# Clone repo
git clone https://github.com/jakainic/belief-credence
cd belief-credence

# Install dependencies
pip install -e . -r requirements.txt

# Test setup (IMPORTANT: run this first!)
python scripts/test_setup.py
```

This quick test (~2-3 min) verifies:
- HF token is configured
- Model loads successfully
- All three methods work
- Outputs can be saved

### 2. Run Evaluation

```bash
python scripts/run_evaluation.py
```

This will:
- Load Llama-2-13b-chat-hf (8-bit quantization)
- Train CCS probe on well-established facts
- Evaluate all 3 methods on contested facts
- Time each method separately
- Save outputs to `outputs/runpod_evaluation/`

**Expected runtime:** ~20-40 minutes depending on GPU (13B is ~2x slower than 7B)

### 3. Generate Plots

```bash
python scripts/generate_plots.py
```

This will:
- Load saved estimates
- Generate 4 comparison plots
- Save to `outputs/visualizations/`

### 4. Download Results

Download the `outputs/` folder from RunPod to view:
- `outputs/runpod_evaluation/*.json` - Raw estimates
- `outputs/visualizations/*.png` - Comparison plots

## Scripts

### `test_setup.py`

Quick validation script to verify everything works before running full evaluation.

**Features:**
- Checks HF token is set
- Loads model (with timing)
- Trains small CCS probe (10 epochs, 2 claims)
- Tests all 3 methods on a single claim
- Verifies output saving works

**Expected runtime:** ~2-3 minutes

**Run this FIRST** to catch any issues before the longer evaluation.

### `run_evaluation.py`

Runs full evaluation pipeline with proper train/validation/test split.

**Features:**
- Creates stratified 60/20/20 train/val/test split with mixed belief types
- Trains CCS probe on training set only
- Evaluates all methods on held-out test set
- Validation set available for hyperparameter tuning
- Times each stage (model loading, CCS training, per-method evaluation)
- Progress indicators with ETA
- Saves estimates to JSON
- Prints summary statistics and split details

**Data Split Strategy:**
- **Training (60%)**: ~54 claims - Mixed belief types for CCS probe training
- **Validation (20%)**: ~18 claims - Mixed belief types for hyperparameter tuning
- **Test (20%)**: ~18 claims - Mixed belief types for final evaluation
- All splits contain a representative mix of:
  - Well-established facts
  - Contested facts
  - Certain predictions
  - Uncertain predictions
  - Normative judgments
  - Metaphysical beliefs

**Important: Fair Comparison**
- All three methods are evaluated on the **EXACT SAME test set**
- CCS trains on training set, DirectPrompting and LogitGap don't need training
- Same test claims ensure fair, apples-to-apples comparison
- Split is deterministic (seed=42) for reproducibility

**Output:**
```
outputs/runpod_evaluation/
├── direct_prompting_Llama-2-13b-chat-hf.json  # Test set results
├── logit_gap_Llama-2-13b-chat-hf.json         # Test set results
├── ccs_Llama-2-13b-chat-hf_layer-1.json       # Test set results
└── split_info.json                            # Documents exact train/val/test split
```

### `generate_plots.py`

Generates comparison visualizations from saved estimates.

**Usage:**
```bash
# Default (compares all methods in outputs/runpod_evaluation/)
python scripts/generate_plots.py

# Compare only chat model results
python scripts/generate_plots.py --pattern "*chat-hf*.json"

# Compare only base model results
python scripts/generate_plots.py --pattern "*7b-hf.json" --report-name base_model

# Custom directories and filtering
python scripts/generate_plots.py \
    --input-dir path/to/estimates \
    --output-dir path/to/plots \
    --pattern "*chat*.json" \
    --report-name my_comparison
```

**Output:**
```
outputs/visualizations/
├── method_comparison_overview.png      # 4-panel comprehensive view
├── method_comparison_claims.png        # Claim-by-claim bars
└── method_comparison_calibration.png   # Distribution curves
```

## Customization

### Evaluate on Different Dataset

Edit `run_evaluation.py`:

```python
# Change evaluation dataset
eval_claim_sets = get_dataset(BeliefType.UNCERTAIN_PREDICTIONS)  # Instead of CONTESTED_FACT
```

Available datasets:
- `WELL_ESTABLISHED_FACT` - Scientific facts (low disagreement expected)
- `CONTESTED_FACT` - Debated claims (high disagreement expected)
- `CERTAIN_PREDICTION` - High-confidence future events
- `UNCERTAIN_PREDICTION` - Speculative future claims
- `NORMATIVE_JUDGMENT` - Moral/political values
- `METAPHYSICAL_BELIEF` - Philosophical positions

### Change CCS Layer

Edit `run_evaluation.py`:

```python
ccs = CCS(model=model, direction_method="logit_gap", layer=-2)  # Try different layer
```

### Use Different Model

Edit `run_evaluation.py`:

```python
model = ModelWrapper("meta-llama/Meta-Llama-3-8B", load_in_8bit=True)
```

## Troubleshooting

**Out of memory?**
- Model is already using 8-bit quantization
- Try reducing batch processing or use smaller model

**HF token not found?**
- Verify: `echo $HF_TOKEN` in RunPod terminal
- Should see your token
- If empty, set in RunPod pod configuration

**Plots not generating?**
- Make sure matplotlib/seaborn are installed: `pip install matplotlib seaborn`
- Check `outputs/runpod_evaluation/` has JSON files

## Expected Timings (A100 GPU)

Approximate times for 3 contested fact claims:

- Model loading: ~30s
- CCS training (3 claims, 100 epochs): ~2-3 min
- Direct Prompting: ~1-2s per claim
- Logit Gap: ~0.5-1s per claim
- CCS evaluation: ~0.5-1s per claim

Total: ~5-10 minutes for full pipeline
