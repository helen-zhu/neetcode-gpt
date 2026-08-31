import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            for layer in model:
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    # A neuron is dead if it is <= 0 for ALL samples in the batch (dim=0)
                    is_dead = (x <= 0).all(dim=0)
                    dead_fraction = is_dead.float().mean().item()
                    stats.append({
                        'mean': round(x.mean().item(), 4),
                        'std': round(x.std().item(), 4),
                        'dead_fraction': round(dead_fraction, 4)
                    })
        return stats


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        stats = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                stats.append({
                    'mean': round(grad.mean().item(), 4),
                    'std': round(grad.std().item(), 4),
                    'norm': round(torch.norm(grad).item(), 4)
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for x in activation_stats:
            if x['dead_fraction'] > 0.5:
                return 'dead_neurons'
        for x in gradient_stats:
            if x['norm'] > 1000:
                return 'exploding_gradients'
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        for x in activation_stats:
            if x['std'] < 0.1:
                return 'vanishing_gradients'
            if x['std'] > 10.0:
                return 'exploding gradients'
        return 'healthy'
        
