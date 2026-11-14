import csv
import json

import pytest

from geoguessr_benchmark.artifacts import _REQUIRED, STAGES, BENCHMARK_ID, atomic_write_bytes, canonical_json_bytes, sha256_file, verify_stage
from geoguessr_benchmark.reporting import build_report


def test_report_recomputes_six_locked_predictions_and_preserves_null_groups(tmp_path):
    paths = set().union(*(_REQUIRED[s] for s in STAGES[:STAGES.index("tested")+1]))
    for path in paths:
        atomic_write_bytes(tmp_path / path, b"fixture")
    atomic_write_bytes(tmp_path/"src/fixture.py",b"x=1\n")
    atomic_write_bytes(tmp_path/"pyproject.toml",b"[project]\n")
    from geoguessr_benchmark.preflight import source_code_sha256
    atomic_write_bytes(tmp_path/"environment.lock.txt",canonical_json_bytes({"code_sha256":source_code_sha256(tmp_path)}))
    atomic_write_bytes(tmp_path / "splits/test.csv", b"image_id,class_id,split,is_synthetic\na,0,test,false\n")
    group_rows = "class_id,class_name,group\n0,A,tail\n" + "".join(f"{i},C{i},many\n" for i in range(1,124))
    atomic_write_bytes(tmp_path / "splits/class_groups.csv", group_rows.encode())
    atomic_write_bytes(tmp_path / "splits/train.csv", b"image_id,is_synthetic\na,false\n")
    atomic_write_bytes(tmp_path / "splits/train_augmented.csv", b"image_id,is_synthetic\na,false\n")
    atomic_write_bytes(tmp_path / "splits/val.csv", b"image_id\na\n")
    atomic_write_bytes(tmp_path / "splits/split_metadata.json", canonical_json_bytes({"evaluable_ids":{"val":[0],"test":[0]}}))
    atomic_write_bytes(tmp_path / "augmentation/manifests/qc_summary.json", canonical_json_bytes({"attempts":3,"accepted":1,"rejected":2}))
    for condition in ("original","augmented"):
        for seed in (42,123,2026):
            base = f"runs/{condition}/seed_{seed}"
            prediction = f'{base}/predictions_test.csv'
            atomic_write_bytes(tmp_path / prediction, b'image_id,true_class_id,predicted_class_id,top5_class_ids,top5_scores\na,0,0,"[0,1,2,3,4]","[5,4,3,2,1]"\n')
            metadata = f"{base}/run_metadata.json"
            atomic_write_bytes(tmp_path / metadata, canonical_json_bytes({"condition":condition,"seed":seed,"duration_seconds":10,"epochs_completed":20,"optimizer_steps":20,"selected_epoch":1,"batch_size":16,"train_count":1,"original_count":1,"synthetic_count":0,"failure_history":[]}))
            paths.add(metadata)
    artifacts = {name:sha256_file(tmp_path / name) for name in paths}
    lock = {"schema_version":1,"benchmark_id":BENCHMARK_ID,"stage":"tested","artifacts":artifacts,
            "test_prediction_inputs":{f"{condition}/seed_{seed}":{"checkpoint_sha256":artifacts[f"runs/{condition}/seed_{seed}/best_checkpoint.pt"], "test_manifest_sha256":artifacts["splits/test.csv"], "config_sha256":artifacts["configs/benchmark.yaml"]} for condition in ("original","augmented") for seed in (42,123,2026)}}
    atomic_write_bytes(tmp_path / "benchmark_lock.json", canonical_json_bytes(lock))
    build_report(tmp_path)
    assert verify_stage(tmp_path,"reported")["stage"] == "reported"
    aggregate = json.loads((tmp_path / "reports/aggregate_metrics.json").read_text(encoding="utf-8"))
    assert aggregate["metrics"]["macro_f1"]["delta"]["mean"] == 0
    assert aggregate["groups"]["many"]["macro_f1"]["original"]["mean"] is None
    text = (tmp_path / "reports/final_report.md").read_text(encoding="utf-8")
    assert "Macro-F1" in text and "124" in text and "20" in text
    from geoguessr_benchmark.__main__ import main
    assert main(["--root", str(tmp_path), "report"]) == 0
    lock = json.loads((tmp_path / "benchmark_lock.json").read_text(encoding="utf-8"))
    path = "runs/original/seed_42/run_metadata.json"
    details = json.loads((tmp_path / path).read_text(encoding="utf-8"))
    details["train_count"] = details["original_count"] = 2
    atomic_write_bytes(tmp_path / path, canonical_json_bytes(details))
    lock["artifacts"][path] = sha256_file(tmp_path / path)
    atomic_write_bytes(tmp_path / "benchmark_lock.json", canonical_json_bytes(lock))
    with pytest.raises(ValueError, match="count"):
        build_report(tmp_path)
