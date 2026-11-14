import pytest

from geoguessr_benchmark.augmentation.policy import plan_candidate


def test_seed_and_round_robin_are_stable():
    sources = [
        {"image_id":"b", "filepath":"data/b.jpg", "class_id":7, "split":"train", "is_synthetic":"false"},
        {"image_id":"a", "filepath":"data/a.jpg", "class_id":7, "split":"train", "is_synthetic":"false"},
    ]
    assert plan_candidate("traditional", 7, 0, sources)["generation_seed"] == 1121161069
    assert plan_candidate("traditional", 7, 1, sources)["generation_seed"] == 3767014528
    assert plan_candidate("da_fusion", 7, 0, sources)["generation_seed"] == 3439134376
    assert plan_candidate("traditional", 7, 0, sources)["source_image_id"] == "a"
    assert plan_candidate("traditional", 7, 1, sources)["source_image_id"] == "b"
    sources[0]["split"] = "test"
    with pytest.raises(ValueError):
        plan_candidate("traditional", 7, 0, sources)


def test_planning_respects_group_budget_attempt_cap_and_resume():
    from geoguessr_benchmark.augmentation.policy import next_candidate
    sources = [{"image_id":"a", "filepath":"data/a.jpg", "class_id":7, "split":"train", "is_synthetic":"false"}]
    group = {"class_id":7, "group":"tail", "required_accepted":2}
    first = next_candidate(group, sources, [])
    assert first["attempt_index"] == 0 and first["augmentation_type"] == "da_fusion"
    attempts = [dict(first, filter_status="rejected", accepted=False)]
    assert next_candidate(group,sources,attempts)["attempt_index"] == 1
    attempts = [dict(next_candidate(group,sources,attempts), filter_status="accepted", accepted=True), *attempts]
    assert next_candidate(group,sources,attempts)["attempt_index"] == 2
    attempts.append(dict(next_candidate(group,sources,attempts), filter_status="accepted", accepted=True))
    assert next_candidate(group,sources,attempts) is None
    assert next_candidate({"class_id":7,"group":"many","required_accepted":0}, sources, []) is None
    capped = [dict(plan_candidate("da_fusion",7,i,sources),filter_status="rejected",accepted=False) for i in range(20)]
    with pytest.raises(ValueError, match="shortage"):
        next_candidate(group,sources,capped)
