"""Debug script to check dtype issues."""

import torch
from belief_credence.model_utils import ModelWrapper
from belief_credence.core import Claim

print("=" * 80)
print("DTYPE DEBUG")
print("=" * 80)

# Load model
print("\nLoading model...")
model = ModelWrapper("meta-llama/Llama-2-7b-hf", load_in_8bit=True)
print(f"Model device: {model.device}")

# Get activations
print("\nGetting activations...")
claim = Claim(
    statement="The Earth orbits around the Sun.",
    negation="The Earth does not orbit around the Sun.",
)

pos_hidden, neg_hidden = model.get_contrast_activations(
    claim.statement, claim.negation, layer=-1
)

print(f"pos_hidden dtype: {pos_hidden.dtype}")
print(f"pos_hidden device: {pos_hidden.device}")
print(f"pos_hidden shape: {pos_hidden.shape}")

# Test mean
pos_mean = pos_hidden.mean(dim=0)
print(f"\npos_mean dtype: {pos_mean.dtype}")
print(f"pos_mean device: {pos_mean.device}")

# Test stacking
activations_list = [pos_mean, pos_mean]
X = torch.stack(activations_list)
print(f"\nStacked X dtype: {X.dtype}")
print(f"Stacked X device: {X.device}")

# Test .to() conversion
X_converted = X.to(device=model.device, dtype=torch.float32)
print(f"\nAfter .to(dtype=torch.float32):")
print(f"  dtype: {X_converted.dtype}")
print(f"  device: {X_converted.device}")

# Test probe
from belief_credence.ccs import CCSProbe

input_dim = X.shape[1]
probe = CCSProbe(input_dim).to(model.device)

print(f"\nProbe linear weight dtype: {probe.linear.weight.dtype}")
print(f"Probe linear weight device: {probe.linear.weight.device}")

# Try forward pass
print("\nTrying forward pass with X_converted (should work)...")
try:
    output = probe(X_converted)
    print(f"✓ Success! Output dtype: {output.dtype}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nTrying forward pass with X (might fail)...")
try:
    output = probe(X)
    print(f"✓ Success! Output dtype: {output.dtype}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 80)
