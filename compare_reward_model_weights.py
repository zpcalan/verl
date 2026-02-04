#!/usr/bin/env python3
"""
Script to compare reward model weights exported from FSDP model with local model weights.

Usage:
    python compare_reward_model_weights.py \
        --exported_weights reward_model_weights_rank_0.pt \
        --local_model_path /path/to/local/model \
        [--tolerance 1e-5]
"""

import argparse
import torch
from transformers import AutoModelForTokenClassification, AutoConfig
import sys


def load_local_model_weights(model_path: str, trust_remote_code: bool = False):
    """Load weights from a local HuggingFace model."""
    print(f"Loading local model from: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    config.num_labels = 1
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float32,  # Use float32 for comparison
    )
    
    # Get state dict and convert to CPU float32
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu().float()
    
    return state_dict, config


def load_exported_weights(export_path: str):
    """Load exported weights from file."""
    print(f"Loading exported weights from: {export_path}")
    state_dict = torch.load(export_path, map_location='cpu')
    
    # Convert to float32 for comparison
    for key in state_dict:
        if isinstance(state_dict[key], torch.Tensor):
            state_dict[key] = state_dict[key].float()
    
    return state_dict


def normalize_key(key: str):
    """Normalize key names to handle potential differences."""
    # Remove common prefixes that might differ
    key = key.replace("_fsdp_wrapped_module.", "")
    key = key.replace("module.", "")
    return key


def compare_weights(exported_dict: dict, local_dict: dict, tolerance: float = 1e-5):
    """Compare two state dictionaries."""
    print("\n" + "="*80)
    print("Comparing weights...")
    print("="*80)
    
    # Normalize keys
    exported_keys = {normalize_key(k): k for k in exported_dict.keys()}
    local_keys = {normalize_key(k): k for k in local_dict.keys()}
    
    exported_normalized = set(exported_keys.keys())
    local_normalized = set(local_keys.keys())
    
    # Find differences
    only_in_exported = exported_normalized - local_normalized
    only_in_local = local_normalized - exported_normalized
    common_keys = exported_normalized & local_normalized
    
    print(f"\nTotal keys in exported: {len(exported_normalized)}")
    print(f"Total keys in local: {len(local_normalized)}")
    print(f"Common keys: {len(common_keys)}")
    
    if only_in_exported:
        print(f"\n⚠️  Keys only in exported ({len(only_in_exported)}):")
        for key in sorted(list(only_in_exported))[:10]:
            print(f"  - {key}")
        if len(only_in_exported) > 10:
            print(f"  ... and {len(only_in_exported) - 10} more")
    
    if only_in_local:
        print(f"\n⚠️  Keys only in local ({len(only_in_local)}):")
        for key in sorted(list(only_in_local))[:10]:
            print(f"  - {key}")
        if len(only_in_local) > 10:
            print(f"  ... and {len(only_in_local) - 10} more")
    
    # Compare common keys
    print(f"\n{'='*80}")
    print("Comparing common keys...")
    print("="*80)
    
    mismatches = []
    max_diff = 0.0
    max_diff_key = None
    
    for norm_key in sorted(common_keys):
        exported_key = exported_keys[norm_key]
        local_key = local_keys[norm_key]
        
        exported_tensor = exported_dict[exported_key]
        local_tensor = local_dict[local_key]
        
        # Check shapes
        if exported_tensor.shape != local_tensor.shape:
            mismatches.append({
                'key': norm_key,
                'issue': 'shape_mismatch',
                'exported_shape': exported_tensor.shape,
                'local_shape': local_tensor.shape,
            })
            continue
        
        # Compare values
        diff = torch.abs(exported_tensor - local_tensor)
        max_diff_val = diff.max().item()
        mean_diff = diff.mean().item()
        
        if max_diff_val > tolerance:
            mismatches.append({
                'key': norm_key,
                'issue': 'value_mismatch',
                'max_diff': max_diff_val,
                'mean_diff': mean_diff,
                'exported_shape': exported_tensor.shape,
            })
        
        if max_diff_val > max_diff:
            max_diff = max_diff_val
            max_diff_key = norm_key
    
    # Print results
    if not mismatches:
        print(f"\n✅ All {len(common_keys)} common keys match within tolerance {tolerance}")
        print(f"   Maximum difference: {max_diff:.2e} (key: {max_diff_key})")
    else:
        print(f"\n❌ Found {len(mismatches)} mismatches:")
        for i, mismatch in enumerate(mismatches[:20]):  # Show first 20
            if mismatch['issue'] == 'shape_mismatch':
                print(f"  {i+1}. {mismatch['key']}:")
                print(f"     Exported shape: {mismatch['exported_shape']}")
                print(f"     Local shape: {mismatch['local_shape']}")
            elif mismatch['issue'] == 'value_mismatch':
                print(f"  {i+1}. {mismatch['key']}:")
                print(f"     Max diff: {mismatch['max_diff']:.2e}")
                print(f"     Mean diff: {mismatch['mean_diff']:.2e}")
                print(f"     Shape: {mismatch['exported_shape']}")
        
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more mismatches")
    
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"  Maximum difference across all keys: {max_diff:.2e}")
    if max_diff_key:
        print(f"  Key with maximum difference: {max_diff_key}")
    print("="*80)
    
    return len(mismatches) == 0, mismatches


def main():
    parser = argparse.ArgumentParser(description="Compare reward model weights")
    parser.add_argument(
        "--exported_weights",
        type=str,
        required=True,
        help="Path to exported weights file (.pt)"
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        required=True,
        help="Path to local HuggingFace model"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for comparing weights (default: 1e-5)"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model"
    )
    
    args = parser.parse_args()
    
    # Load weights
    try:
        exported_dict = load_exported_weights(args.exported_weights)
    except Exception as e:
        print(f"❌ Error loading exported weights: {e}")
        sys.exit(1)
    
    try:
        local_dict, config = load_local_model_weights(args.local_model_path, args.trust_remote_code)
    except Exception as e:
        print(f"❌ Error loading local model: {e}")
        sys.exit(1)
    
    # Compare
    match, mismatches = compare_weights(exported_dict, local_dict, args.tolerance)
    
    if match:
        print("\n✅ Weights match! Reward model is correctly loaded.")
        sys.exit(0)
    else:
        print("\n❌ Weights do not match. Please check the mismatches above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

