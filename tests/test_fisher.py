from __future__ import annotations

import math

import pytest
import torch

from corerank_synth.fisher import rank_score_from_K


def test_rank_score_zero_information_is_not_full_rank() -> None:
    K = torch.zeros(2, 3, 3)

    score, diag = rank_score_from_K(K, eps=1e-3)

    assert score.item() == pytest.approx(3 * math.log(1e-3), rel=1e-6)
    assert torch.allclose(diag["effective_rank"], torch.zeros(2))
    assert torch.allclose(diag["trace"], torch.zeros(2))


def test_rank_score_separates_rank_one_from_full_rank() -> None:
    K_rank_one = torch.diag(torch.tensor([1.0, 0.0, 0.0])).unsqueeze(0)
    K_full = torch.eye(3).unsqueeze(0)

    score_rank_one, diag_rank_one = rank_score_from_K(K_rank_one, eps=1e-3)
    score_full, diag_full = rank_score_from_K(K_full, eps=1e-3)

    assert score_full.item() > score_rank_one.item()
    assert diag_rank_one["effective_rank"].item() == pytest.approx(1.0, rel=1e-5)
    assert diag_full["effective_rank"].item() == pytest.approx(3.0, rel=1e-5)


def test_rank_score_rejects_non_square_inputs() -> None:
    with pytest.raises(ValueError):
        rank_score_from_K(torch.zeros(2, 3, 4))
