#!/usr/bin/env python3
"""
Simple test script to verify that the PPO configuration fix works.
"""

import sys
import os
import torch.nn as nn

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_config import PPO_CONFIG

def test_activation_function_conversion():
    """Test the activation function string to function conversion."""
    
    # Test the conversion logic
    ppo_params = PPO_CONFIG.copy()
    
    # Convert activation function string to proper function
    if "policy_kwargs" in ppo_params and "activation_fn" in ppo_params["policy_kwargs"]:
        activation_str = ppo_params["policy_kwargs"]["activation_fn"]
        print(f"Original activation function: {activation_str} (type: {type(activation_str)})")
        
        if activation_str == "relu":
            ppo_params["policy_kwargs"]["activation_fn"] = nn.ReLU
        elif activation_str == "tanh":
            ppo_params["policy_kwargs"]["activation_fn"] = nn.Tanh
        elif activation_str == "sigmoid":
            ppo_params["policy_kwargs"]["activation_fn"] = nn.Sigmoid
        else:
            print(f"Warning: Unknown activation function '{activation_str}', using ReLU")
            ppo_params["policy_kwargs"]["activation_fn"] = nn.ReLU
        
        converted_fn = ppo_params["policy_kwargs"]["activation_fn"]
        print(f"Converted activation function: {converted_fn} (type: {type(converted_fn)})")
        
        # Test that it's callable
        try:
            activation_layer = converted_fn()
            print(f"Successfully created activation layer: {activation_layer}")
            return True
        except Exception as e:
            print(f"Error creating activation layer: {e}")
            return False
    
    return False

def test_net_arch_format():
    """Test that the network architecture format is correct."""
    
    ppo_params = PPO_CONFIG.copy()
    
    if "policy_kwargs" in ppo_params and "net_arch" in ppo_params["policy_kwargs"]:
        net_arch = ppo_params["policy_kwargs"]["net_arch"]
        print(f"Network architecture: {net_arch} (type: {type(net_arch)})")
        
        # Check if it's the correct format (dict, not list of dict)
        if isinstance(net_arch, dict):
            print("✓ Network architecture format is correct (dict)")
            return True
        elif isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            print("⚠ Network architecture format is deprecated (list of dict)")
            return False
        else:
            print("✗ Network architecture format is incorrect")
            return False
    
    return False

if __name__ == "__main__":
    print("Testing PPO configuration fixes...")
    print("=" * 50)
    
    # Test activation function conversion
    print("\n1. Testing activation function conversion:")
    activation_ok = test_activation_function_conversion()
    
    # Test network architecture format
    print("\n2. Testing network architecture format:")
    net_arch_ok = test_net_arch_format()
    
    print("\n" + "=" * 50)
    if activation_ok and net_arch_ok:
        print("✓ All tests passed! The configuration should work correctly.")
    else:
        print("✗ Some tests failed. Please check the configuration.")
    
    print("\nTest completed.")
