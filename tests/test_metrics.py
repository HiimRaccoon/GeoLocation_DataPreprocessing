import pytest

from geoguessr_benchmark.metrics import evaluate_predictions, rank_logits


def test_full_confusion_and_absent_class_metrics():
    rows = [{"true_id": a, "pred_id": b, "top5_ids": [b] + [i for i in range(5) if i != b][:4]} for a,b in [(0,0),(0,2),(1,3),(2,0)]]
    result = evaluate_predictions(rows, [0,1,2], {"g": [0,1], "empty": [3]})
    assert result["metrics"]["top1_accuracy"] == .25
    assert result["metrics"]["macro_f1"] == pytest.approx(1/6)
    assert result["groups"]["g"]["macro_f1"] == .25
    assert result["groups"]["g"]["macro_recall"] == .25
    assert result["groups"]["empty"]["macro_f1"] is None
    assert result["per_class"][3]["f1"] is None
    assert result["confusion"][2][0] == 1


def test_top5_ties_break_by_class_id():
    assert rank_logits([1.0] * 124)[:5] == [0,1,2,3,4]
    result = evaluate_predictions([{"true_id": 5, "pred_id": 0, "top5_ids": rank_logits([1.0] * 124)[:5]}], [5], {})
    assert result["metrics"]["top5_accuracy"] == 0


