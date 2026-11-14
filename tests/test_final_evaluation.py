import pytest

from geoguessr_benchmark.evaluate import validate_predictions


def test_predictions_validate_ids_labels_and_full_top5_order():
    manifest = [{"image_id":"a", "class_id":"3"}, {"image_id":"b", "class_id":"5"}]
    predictions = [
        {"image_id":"b", "true_class_id":5, "predicted_class_id":0, "top5_class_ids":[0,1,2,3,4], "top5_scores":[5.,4.,3.,2.,1.]},
        {"image_id":"a", "true_class_id":3, "predicted_class_id":0, "top5_class_ids":[0,1,2,3,4], "top5_scores":[5.,5.,3.,2.,1.]},
    ]
    result = validate_predictions(manifest, predictions)
    assert len(result) == 2
    with pytest.raises(ValueError):
        validate_predictions(manifest, predictions[:1])
    broken = [dict(row) for row in predictions]
    broken[1]["true_class_id"] = 2
    with pytest.raises(ValueError):
        validate_predictions(manifest, broken)
    broken[1]["true_class_id"] = 3
    broken[1]["top5_class_ids"] = [1,0,2,3,4]
    broken[1]["predicted_class_id"] = 1
    with pytest.raises(ValueError):
        validate_predictions(manifest, broken)


def test_loading_test_predictions_requires_tested_lock_and_checks_hashes(tmp_path):
    import csv
    from geoguessr_benchmark.artifacts import _REQUIRED, STAGES, BENCHMARK_ID, atomic_write_bytes, canonical_json_bytes, sha256_file
    from geoguessr_benchmark.evaluate import load_test_predictions
    paths = set().union(*(_REQUIRED[stage] for stage in STAGES[:STAGES.index("trained")+1]))
    for relative in paths:
        atomic_write_bytes(tmp_path / relative, b"fixture")
    test_csv = b"image_id,class_id,split,is_synthetic\na,3,test,false\n"
    atomic_write_bytes(tmp_path / "splits/test.csv", test_csv)
    artifacts = {name: sha256_file(tmp_path / name) for name in paths}
    lock = {"schema_version":1,"benchmark_id":BENCHMARK_ID,"stage":"trained","artifacts":artifacts}
    atomic_write_bytes(tmp_path / "benchmark_lock.json", canonical_json_bytes(lock))
    with pytest.raises(ValueError):
        load_test_predictions(tmp_path, "original", 42)
    for condition in ("original","augmented"):
        for seed in (42,123,2026):
            relative = f"runs/{condition}/seed_{seed}/predictions_test.csv"
            payload = 'image_id,true_class_id,predicted_class_id,top5_class_ids,top5_scores\na,3,0,"[0,1,2,3,4]","[5,4,3,2,1]"\n'.encode()
            atomic_write_bytes(tmp_path / relative, payload)
            artifacts[relative] = sha256_file(tmp_path / relative)
    lock["stage"] = "tested"
    lock["test_prediction_inputs"] = {
        f"{condition}/seed_{seed}": {
            "checkpoint_sha256": artifacts[f"runs/{condition}/seed_{seed}/best_checkpoint.pt"],
            "test_manifest_sha256": artifacts["splits/test.csv"],
            "config_sha256": artifacts["configs/benchmark.yaml"],
        } for condition in ("original", "augmented") for seed in (42, 123, 2026)
    }
    atomic_write_bytes(tmp_path / "benchmark_lock.json", canonical_json_bytes(lock))
    assert load_test_predictions(tmp_path,"original",42)[0]["true_id"] == 3
    (tmp_path / "runs/original/seed_42/predictions_test.csv").write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        load_test_predictions(tmp_path,"original",42)


def test_final_evaluation_is_gated_and_reuses_locked_predictions_after_failure(tmp_path, monkeypatch):
    import json
    import geoguessr_benchmark.evaluate as evaluation
    from geoguessr_benchmark.artifacts import _REQUIRED, STAGES, BENCHMARK_ID, atomic_write_bytes, canonical_json_bytes, sha256_file, verify_stage
    paths=set().union(*(_REQUIRED[s] for s in STAGES[:STAGES.index("trained")+1]))
    for name in paths: atomic_write_bytes(tmp_path/name,b"fixture")
    atomic_write_bytes(tmp_path/"splits/test.csv",b"image_id,class_id,split,is_synthetic\na,0,test,false\n")
    atomic_write_bytes(tmp_path/"splits/split_metadata.json",canonical_json_bytes({"evaluable_ids":{"test":[0],"val":[0]}}))
    groups="class_id,class_name,group\n0,A,tail\n"+"".join(f"{i},C{i},many\n" for i in range(1,124))
    atomic_write_bytes(tmp_path/"splits/class_groups.csv",groups.encode())
    for condition in ("original","augmented"):
        for seed in (42,123,2026):
            base=f"runs/{condition}/seed_{seed}"
            for file,value in (("run_metadata.json",{"condition":condition,"seed":seed}), ("metrics.json",{"validation_at_best":{"macro_f1":.5}}), ("group_metrics.json",{"validation_at_best":{}})):
                name=f"{base}/{file}"
                atomic_write_bytes(tmp_path/name,canonical_json_bytes(value))
                paths.add(name)
    artifacts={name:sha256_file(tmp_path/name) for name in paths}
    lock={"schema_version":1,"benchmark_id":BENCHMARK_ID,"stage":"initialized","artifacts":artifacts}
    atomic_write_bytes(tmp_path/"benchmark_lock.json",canonical_json_bytes(lock))
    calls=[]
    def infer(root,condition,seed,lock,manifest):
        calls.append((condition,seed))
        if len(calls)==2:
            raise OSError("infrastructure failed")
        return [{"image_id":"a","true_class_id":0,"predicted_class_id":0,"top5_class_ids":[0,1,2,3,4],"top5_scores":[5.,4.,3.,2.,1.]}]
    monkeypatch.setattr(evaluation,"_run_test_inference",infer)
    with pytest.raises(ValueError): evaluation.evaluate_test_once(tmp_path)
    assert not calls
    lock["stage"]="trained"
    atomic_write_bytes(tmp_path/"benchmark_lock.json",canonical_json_bytes(lock))
    import geoguessr_benchmark.preflight as preflight
    monkeypatch.setattr(preflight,"_configure_determinism",lambda:None)
    def mismatch(root,lock): raise ValueError("runtime mismatch")
    monkeypatch.setattr(preflight,"verify_runtime",mismatch)
    with pytest.raises(ValueError,match="runtime mismatch"):
        evaluation.evaluate_test_once(tmp_path)
    assert not calls
    monkeypatch.setattr(preflight,"verify_runtime",lambda root,lock:None)
    import geoguessr_benchmark.artifacts as artifacts_module
    real_write=artifacts_module.atomic_write_bytes
    def interrupted(path,payload):
        if str(path).endswith("predictions_test.lock.json"):
            raise OSError("publication interrupted")
        return real_write(path,payload)
    monkeypatch.setattr(artifacts_module,"atomic_write_bytes",interrupted)
    with pytest.raises(OSError,match="publication"):
        evaluation.evaluate_test_once(tmp_path)
    monkeypatch.setattr(artifacts_module,"atomic_write_bytes",real_write)
    with pytest.raises(OSError,match="infrastructure"): evaluation.evaluate_test_once(tmp_path)
    assert calls==[("original",42),("augmented",42)]
    assert (tmp_path/"runs/original/seed_42/predictions_test.csv").exists()
    from geoguessr_benchmark.__main__ import main
    assert main(["--root",str(tmp_path),"test-all"]) == 0
    result=json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))
    assert result["stage"]=="tested" and verify_stage(tmp_path,"tested")
    assert calls.count(("original",42))==1
    assert len(evaluation.load_test_predictions(tmp_path,"original",42))==1
    assert evaluation.evaluate_test_once(tmp_path)["stage"]=="tested"
    assert len(calls)==7
