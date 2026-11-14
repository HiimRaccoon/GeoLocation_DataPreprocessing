import pytest

from geoguessr_benchmark.reporting import aggregate_paired


def test_paired_statistics_match_seed_not_input_order():
    a = [{"seed": 2026, "value": .6}, {"seed":42, "value":.2}, {"seed":123, "value":.4}]
    b = [{"seed": 123, "value": .6}, {"seed":2026, "value":.7}, {"seed":42, "value":.3}]
    result = aggregate_paired(a, b)
    assert result["original"]["mean"] == pytest.approx(.4)
    assert result["original"]["std"] == pytest.approx(.2)
    assert result["augmented"]["mean"] == pytest.approx(.5333333333333333)
    assert result["delta"]["mean"] == pytest.approx(.1333333333333333)
    assert result["delta"]["std"] == pytest.approx(.0577350269)
    assert result["delta_pp"]["mean"] == pytest.approx(13.333333333333)
    with pytest.raises(ValueError):
        aggregate_paired(a, b[:-1])
    with pytest.raises(ValueError):
        aggregate_paired(a, b[:2] + b[:1])


def test_empty_group_null_is_preserved():
    a = [{"seed":s, "value":None} for s in (42,123,2026)]
    assert aggregate_paired(a,a)["delta"]["mean"] is None
