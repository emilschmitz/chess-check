"""Tests for chess loss calculation."""
import numpy as np
import pytest

from src.evaluation.chess_loss import cross_entropy_loss, kl_divergence


def test_cross_entropy_identical_distributions():
    """Test CE loss is minimal for identical distributions."""
    dist = {"e4": 0.5, "d4": 0.3, "Nf3": 0.2}
    loss = cross_entropy_loss(dist, dist)
    # CE of identical distributions should be the entropy
    assert loss > 0
    assert loss < 2.0  # Reasonable bound


def test_cross_entropy_uniform_vs_peaked():
    """Test CE loss is high when distributions differ."""
    uniform = {"e4": 0.333, "d4": 0.333, "Nf3": 0.334}
    peaked = {"e4": 0.9, "d4": 0.05, "Nf3": 0.05}

    loss = cross_entropy_loss(uniform, peaked)
    assert loss > 1.0  # Should be high due to mismatch


def test_cross_entropy_properties():
    """Test mathematical properties of CE loss."""
    dist1 = {"e4": 0.7, "d4": 0.3}
    dist2 = {"e4": 0.4, "d4": 0.6}

    loss = cross_entropy_loss(dist1, dist2)

    # CE loss should be non-negative
    assert loss >= 0

    # CE should be symmetric in inputs? No! CE is not symmetric
    loss_reverse = cross_entropy_loss(dist2, dist1)
    # They should be different
    assert loss != loss_reverse


def test_kl_divergence_identical():
    """Test KL divergence is 0 for identical distributions."""
    dist = {"e4": 0.5, "d4": 0.3, "Nf3": 0.2}
    kl = kl_divergence(dist, dist)
    assert kl < 1e-6  # Should be very close to 0


def test_kl_divergence_positive():
    """Test KL divergence is always positive."""
    dist1 = {"e4": 0.7, "d4": 0.3}
    dist2 = {"e4": 0.4, "d4": 0.6}

    kl = kl_divergence(dist1, dist2)
    assert kl >= 0


def test_loss_with_missing_moves():
    """Test loss calculation handles missing moves in one distribution."""
    predicted = {"e4": 0.5, "d4": 0.3, "Nf3": 0.2}
    target = {"e4": 0.6, "d4": 0.4}  # Missing Nf3

    # Should not crash
    loss = cross_entropy_loss(predicted, target)
    assert loss >= 0
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_probability_normalization():
    """Test that unnormalized probabilities are handled correctly."""
    # Even if probs don't sum to 1, loss should be calculable
    predicted = {"e4": 0.5, "d4": 0.3}  # Sum = 0.8
    target = {"e4": 0.7, "d4": 0.3}     # Sum = 1.0

    loss = cross_entropy_loss(predicted, target)
    # Should return a finite value
    assert not np.isnan(loss)
    assert not np.isinf(loss)
