import pytest
from geoguessr_benchmark.training import learning_rate_at_step


def test_lr_warmup_and_cosine_equal_epoch_progress():
    config = {"learning_rate": 0.0003, "minimum_learning_rate": 0.000001, "epochs": 20}
    assert learning_rate_at_step(0, 100, config) == pytest.approx(3e-5)
    assert learning_rate_at_step(100, 100, config) == pytest.approx(3e-4)
    assert learning_rate_at_step(1050, 100, config) == pytest.approx(0.0001505)
    assert learning_rate_at_step(1999, 100, config) > 1e-6
    assert learning_rate_at_step(21, 2, config) == pytest.approx(0.0001505)



def test_epoch_shuffle_visits_each_record_once_and_rng_resume():
    import torch
    from geoguessr_benchmark.training import epoch_permutation, worker_seed
    generator = torch.Generator().manual_seed(42)
    epoch_permutation(13, generator)
    saved = generator.get_state().clone()
    expected = epoch_permutation(13, generator)
    resumed = torch.Generator().set_state(saved)
    assert epoch_permutation(13, resumed) == expected
    assert sorted(expected) == list(range(13))
    assert worker_seed(2**32 + 13) == 13
