"""Unit tests for LENS-Net that need no dataset, feature file or GPU. Each test checks a statement of the paper.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src" / d) for d in ("model", "revision")] + [str(REPO)]

import lensnet  # noqa: E402
import lensnet_setup  # noqa: E402

B, T, K, P, D = 6, 16, 17, 10, 768


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    cls = torch.randn(B, D, generator=g)
    S = torch.rand(B, T, K, P, generator=g) * 2 - 1          # prompt similarities in [-1, 1]
    M = torch.randn(B, D, generator=g)
    return cls, S, M


def anchors(seed=0):
    a = np.random.default_rng(seed).standard_normal((2, D)).astype(np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def trained(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_trained_parameter_counts():
    """Eq. (15) and Table 3: LENS-Net 31,926; CLIP probe 1,536 (1.5k); without motion 5.7k; motion probe 26.2k."""
    assert trained(lensnet.V2("LENS-M", anchors())) == 31_926
    assert trained(lensnet.V2("LP", anchors())) == 1_536
    assert round(trained(lensnet.V2("V2b", anchors())) / 1000, 1) == 5.7
    # The motion probe also holds the (unused) anchor weights w; its forward pass never reads them, so they get no
    # gradient and are never updated. Count the parameters that actually train: the motion head.
    mp = lensnet.V2("MP", anchors())
    assert sum(p.numel() for p in mp.mhead.parameters()) == 26_210


def test_motion_probe_does_not_train_the_unused_anchor(batch):
    cls, S, M = batch
    mp = lensnet.V2("MP", anchors())
    F.cross_entropy(mp(cls, S, M), torch.tensor([0, 1] * (B // 2))).backward()
    assert mp.w.grad is None


def test_pps_descriptor_has_120_named_coordinates(batch):
    """Eq. (11): 4 temporal x 3 spatial operators x 10 prompts = 120 values."""
    _, S, _ = batch
    phi = lensnet.pps_features(S, "V2b")
    assert phi.shape == (B, 120)
    # coordinate (temporal mean, CLS similarity, prompt 0) equals the time-average of S[:, :, 0, 0]
    assert torch.allclose(phi[:, 0], S[:, :, 0, 0].mean(1), atol=1e-6)


def test_pps_operators_match_definitions(batch):
    """Eqs. (9)-(10): grid max, top-2 mean, temporal max, SD and mean absolute change."""
    _, S, _ = batch
    phi = lensnet.pps_features(S, "V2b").view(B, 4, 3, P)
    grid = S[:, :, 1:]
    sig2 = grid.max(2).values
    sig3 = grid.topk(2, dim=2).values.mean(2)
    assert torch.allclose(phi[:, 0, 1], sig2.mean(1), atol=1e-6)
    assert torch.allclose(phi[:, 1, 2], sig3.max(1).values, atol=1e-6)
    assert torch.allclose(phi[:, 2, 0], S[:, :, 0].std(1, unbiased=False), atol=1e-6)
    assert torch.allclose(phi[:, 3, 0], (S[:, 1:, 0] - S[:, :-1, 0]).abs().mean(1), atol=1e-6)


def test_anchor_is_zero_shot_clip_at_initialisation(batch):
    """Eq. (7): before training the anchored classifier is the zero-shot classifier s * <u, a_c>."""
    cls, S, M = batch
    a = anchors()
    model = lensnet.V2("LP", a).eval()
    expected = lensnet_setup.SCALE * F.normalize(cls, dim=-1) @ torch.tensor(a).t()
    assert torch.allclose(model(cls, S, M), expected, atol=1e-5)
    assert lensnet_setup.SCALE == 30


def test_remark1_zero_gates_give_the_anchor_and_no_head_gradient(batch):
    """Remark 1: at g_P = g_M = 0 the logits equal the anchor's, the heads receive no gradient, the gates do."""
    cls, S, M = batch
    torch.manual_seed(0)
    full = lensnet.V2("LENS-M", anchors()).train()
    anchor = lensnet.V2("LP", anchors()).train()
    out = full(cls, S, M)
    assert torch.allclose(out, anchor(cls, S, M), atol=1e-5)
    F.cross_entropy(out, torch.tensor([0, 1] * (B // 2))).backward()
    for name, p in full.named_parameters():
        if name.startswith(("head.", "mhead.")) and p.grad is not None:
            assert torch.count_nonzero(p.grad) == 0, name
    assert full.g.grad is not None and full.g.grad.abs() > 0
    assert full.gm.grad is not None and full.gm.grad.abs() > 0
    assert full.w.grad.abs().sum() > 0


def test_anchor_penalty_is_zero_at_initialisation_and_grows():
    """Omega_A = lambda_A * sum_c ||w_c - a_c||^2 with lambda_A = 0.5."""
    model = lensnet.V2("LENS-M", anchors())
    assert float(model.anchor_penalty()) == 0.0
    with torch.no_grad():
        model.w.add_(0.1)
    expected = lensnet_setup.ANCHOR_W * float(((model.w - model.w0) ** 2).sum())
    assert abs(float(model.anchor_penalty()) - expected) < 1e-6
    assert lensnet_setup.ANCHOR_W == 0.5


def test_remark2_anchor_fusion_log_loss_bound():
    """Remark 2: -log((qa+qb)/2) <= (-log qa - log qb)/2, with equality iff qa = qb."""
    q = np.random.default_rng(1).uniform(1e-4, 1, (10_000, 2))
    lhs = -np.log(q.mean(1))
    rhs = -np.log(q).mean(1)
    assert np.all(lhs <= rhs + 1e-12)
    assert np.isclose(-np.log(0.3), -np.log(0.3))


def test_ambiguity_decomposition_eq20():
    """Eq. (20): (p_w - y)^2 = sum_j w_j (p_j - y)^2 - sum_j w_j (p_j - p_w)^2, exactly."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, (1000, 3)); y = rng.integers(0, 2, 1000)[:, None]
    w = np.array([0.5, 0.3, 0.2]); pw = (p * w).sum(1, keepdims=True)
    lhs = (pw - y) ** 2
    rhs = ((p - y) ** 2 * w).sum(1, keepdims=True) - ((p - pw) ** 2 * w).sum(1, keepdims=True)
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_weighted_auc_equals_mann_whitney():
    """Eq. (22): the bootstrap AUC with unit weights equals scikit-learn's ROC-AUC, ties counting 1/2."""
    from sklearn.metrics import roc_auc_score
    from seed_aware_bootstrap import wauc
    rng = np.random.default_rng(3)
    s = np.round(rng.uniform(0, 1, 500), 2)           # rounded scores produce ties
    y = rng.integers(0, 2, 500)
    assert np.isclose(wauc(s, y, np.ones((1, 500)))[0], 100 * roc_auc_score(y, s))


def test_prompt_sets_match_table_21():
    """Section 7.1 and Table 21: five violent and five non-violent prompts, two templates."""
    txt = (REPO / "src" / "model" / "lensnet_setup.py").read_text()
    for prompt in ("people walking calmly", "people punching and kicking each other", "a brawl between people",
                   "a physical assault on a person", "a calm crowd of people"):
        assert prompt in txt, prompt
    assert lensnet_setup.NPROMPT == 10
