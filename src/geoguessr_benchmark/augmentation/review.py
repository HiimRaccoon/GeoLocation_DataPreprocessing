"""Offline diffusion review queue and hash-bound decision import."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import uuid

from ..artifacts import BENCHMARK_ID, _resolve_artifact, atomic_write_bytes, canonical_jsonl_bytes, resolve_cleaned_image, sha256_file, verify_stage

RUBRIC = "geographic-cues-v1.2"
REASONS = {"geographic_cue_changed", "generation_artifact", "reject_ambiguous"}


def _jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _train(root):
    with (root / "splits/train.csv").open(encoding="utf-8", newline="") as stream:
        return {row["image_id"]: row for row in csv.DictReader(stream)}


def _holdout_hashes(root):
    hashes, pixels = set(), set()
    for split in ("val", "test"):
        with (root / f"splits/{split}.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                hashes.add(row["sha256"])
                pixels.add(row["decoded_pixel_sha256"])
    return hashes, pixels


def _validated_pair(root, row, train, holdout):
    from ..images import decode_rgb, pixel_sha256
    src = train.get(row["source_image_id"])
    if not src or src["filepath"] != row["source_image_path"] or src["sha256"] != row["source_sha256"] or src["class_name"] != row["class_name"]:
        raise ValueError("source is not the original train image")
    if row.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("incompatible candidate benchmark identity")
    source = resolve_cleaned_image(root, row["source_image_path"])
    output = _resolve_artifact(root, row["synthetic_path"])
    sid = row.get("synthetic_id")
    if not isinstance(sid, str) or not re.fullmatch("[0-9a-f]{64}", sid):
        raise ValueError("invalid synthetic ID")
    expected_path = f'data/augmented_dataset/da_fusion/{src["class_name"]}/{sid}.jpg'
    if row["synthetic_path"] != expected_path:
        raise ValueError("review output is outside the generated dataset")
    if not source.is_file() or not output.is_file() or sha256_file(source) != row["source_sha256"] or sha256_file(output) != row["output_sha256"]:
        raise ValueError("review image hash mismatch")
    if row["output_sha256"] in holdout[0] or pixel_sha256(decode_rgb(output)) in holdout[1]:
        raise ValueError("holdout collision cannot be exported for review")
    return source, output


def export_review_queue(root, output):
    root = Path(root).resolve()
    lock = verify_stage(root, "splits")
    train = _train(root)
    holdout = _holdout_hashes(root)
    rows = _jsonl(root / "augmentation/manifests/candidates.jsonl")
    if any(row.get("benchmark_id") != BENCHMARK_ID for row in rows):
        raise ValueError("incompatible candidate benchmark identity")
    qc_context = None
    if lock["stage"] != "splits":
        from ..config import load_config
        from .qc import build_qc_index, check_candidate
        qc_config = load_config(root)["da_fusion"]
        qc_context = build_qc_index(root)
        accepted_files = {row["output_sha256"] for row in rows if row.get("accepted") is True}
        accepted_pixels = {row["output_pixel_sha256"] for row in rows if row.get("accepted") is True}
    decisions = _jsonl(root / "augmentation/manifests/reviews.jsonl")
    if any(row.get("rubric_version") != RUBRIC for row in decisions):
        raise ValueError("incompatible review version")
    reviewed = {row["synthetic_id"] for row in decisions}
    queue = []
    pairs = []
    for row in rows:
        if row.get("augmentation_type") != "da_fusion" or row.get("filter_status") != "pending" or row.get("qc_passed") is not True or row["synthetic_id"] in reviewed:
            continue
        sid = row["synthetic_id"]
        if not isinstance(sid, str) or not re.fullmatch("[0-9a-f]{64}", sid) or any(x["synthetic_id"] == sid for x in queue):
            raise ValueError("invalid or duplicate synthetic ID")
        try:
            source, generated = _validated_pair(root, row, train, holdout)
        except ValueError as error:
            if "holdout collision" in str(error):
                from ..artifacts import canonical_json_bytes
                atomic_write_bytes(root / "augmentation/manifests/holdout_incident.json", canonical_json_bytes({"reason":"holdout_collision"}))
            raise
        if qc_context is not None:
            qc = check_candidate(root, row, train[row["source_image_id"]], qc_config,
                                 {"context":qc_context,"accepted_file":accepted_files,"accepted_pixel":accepted_pixels})
            if not qc["passed"]:
                raise ValueError(f'review candidate failed automatic QC: {qc["reason"]}')
        queue.append({"synthetic_id": sid, "attempt_index": row["attempt_index"], "source_image_id": row["source_image_id"],
                      "source_image_path": row["source_image_path"], "synthetic_path": row["synthetic_path"],
                      "source_sha256": row["source_sha256"], "output_sha256": row["output_sha256"],
                      "class_name": row["class_name"], "rubric_version": RUBRIC})
        pairs.append((sid, source, generated))
    base = root / "augmentation"
    base.mkdir(parents=True, exist_ok=True)
    temp = base / (".review-staging-" + uuid.uuid4().hex)
    temp.mkdir()
    try:
        (temp / "sources").mkdir()
        (temp / "generated").mkdir()
        for sid, source, generated in pairs:
            shutil.copyfile(source, temp / "sources" / (sid + ".jpg"))
            shutil.copyfile(generated, temp / "generated" / (sid + ".jpg"))
        payload = canonical_jsonl_bytes(queue)
        atomic_write_bytes(temp / "review_queue.jsonl", payload)
        embedded = json.dumps(queue, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        atomic_write_bytes(temp / "index.html", _HTML.replace("__ROWS__", embedded).encode("utf-8"))
        target = base / "review"
        if target.exists():
            target.rename(base / ("review_previous_" + uuid.uuid4().hex))
        temp.rename(target)
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = root / output_path
        atomic_write_bytes(output_path, payload)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return queue


def import_review_decisions(root, input):
    root = Path(root).resolve()
    verify_stage(root, "splits")
    queue = _jsonl(root / "augmentation/manifests/review_queue.jsonl")
    entries = {row["synthetic_id"]: row for row in queue}
    if len(entries) != len(queue):
        raise ValueError("duplicate review queue IDs")
    candidates = {row["synthetic_id"]: row for row in _jsonl(root / "augmentation/manifests/candidates.jsonl")}
    train = _train(root)
    holdout = _holdout_hashes(root)
    existing_path = root / "augmentation/manifests/reviews.jsonl"
    existing = _jsonl(existing_path)
    if any(row.get("rubric_version") != RUBRIC for row in existing + queue):
        raise ValueError("incompatible review version")
    decisions = {row["synthetic_id"]: row for row in existing}
    if len(decisions) != len(existing):
        raise ValueError("duplicate saved review decisions")
    input_path = Path(input)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not input_path.is_file():
        raise ValueError("review input file is missing")
    incoming = _jsonl(input_path)
    seen = set()
    for decision in incoming:
        sid = decision.get("synthetic_id")
        if sid in seen or sid not in entries or sid not in candidates:
            raise ValueError("unknown or duplicate review decision")
        seen.add(sid)
        row = entries[sid]
        candidate = candidates[sid]
        _validated_pair(root, candidate, train, holdout)
        for key in ("source_sha256", "output_sha256"):
            if decision.get(key) != row.get(key) or row.get(key) != candidate.get(key):
                raise ValueError("review hash mismatch")
        if decision.get("rubric_version") != RUBRIC or row.get("rubric_version") != RUBRIC or not decision.get("reviewer_id"):
            raise ValueError("invalid review rubric or reviewer")
        try:
            timestamp = datetime.fromisoformat(decision["review_timestamp"].replace("Z", "+00:00"))
            if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
                raise ValueError("non-UTC timestamp")
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("invalid UTC review timestamp") from error
        choice, reason = decision.get("decision"), decision.get("reason")
        if not ((choice == "accept" and reason in (None, "")) or (choice == "reject" and reason in REASONS)):
            raise ValueError("invalid review decision or reason")
        if sid in decisions and decisions[sid] != decision:
            raise ValueError("conflicting review decision")
        decisions[sid] = decision
    atomic_write_bytes(existing_path, canonical_jsonl_bytes([decisions[sid] for sid in sorted(decisions)]))
    return incoming


_HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Geographic cue review</title>
<style>body{font:16px sans-serif;max-width:1200px;margin:auto;padding:1rem}article{border:1px solid #bbb;padding:1rem;margin:1rem 0}.pair{display:flex;gap:1rem}.pair img{width:49%;object-fit:contain}code{overflow-wrap:anywhere}button{margin:.2rem}</style>
<h1>Geographic cue review</h1><p>Compare source and generated image: road geometry, traffic side, markings, signs and text, buildings, poles, vegetation, vehicles, and other geographic cues. Reject invented or changed cues and severe artifacts; reject ambiguous cases.</p>
<label>Reviewer ID <input id="reviewer"></label><button id="save">Download reviews.jsonl</button><main id="items"></main>
<script>const rows=__ROWS__, decisions={}; const container=document.getElementById('items');
for(const row of rows){const a=document.createElement('article');
const title=document.createElement('h2');title.textContent=row.class_name+' — attempt '+row.attempt_index;a.append(title);
for(const v of ['synthetic_id','source_sha256','output_sha256','rubric_version']){const p=document.createElement('p');p.textContent=v+': '+row[v];a.append(p)}
const pair=document.createElement('div');pair.className='pair';for(const folder of ['sources','generated']){const img=document.createElement('img');img.src=folder+'/'+row.synthetic_id+'.jpg';img.alt=folder;pair.append(img)}a.append(pair);
const state=document.createElement('p');state.textContent='Pending';a.append(state);
for(const [label,decision,reason] of [['Accept','accept',null],['Reject — geographic cue changed','reject','geographic_cue_changed'],['Reject — generation artifact','reject','generation_artifact'],['Reject — ambiguous','reject','reject_ambiguous']]){const b=document.createElement('button');b.textContent=label;b.onclick=()=>{decisions[row.synthetic_id]={...row,decision,reason};state.textContent=label};a.append(b)}container.append(a)}
document.getElementById('save').onclick=()=>{const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('Enter reviewer ID');return}const records=Object.values(decisions).map(x=>({synthetic_id:x.synthetic_id,source_sha256:x.source_sha256,output_sha256:x.output_sha256,reviewer_id:reviewer,review_timestamp:new Date().toISOString(),rubric_version:x.rubric_version,decision:x.decision,reason:x.reason}));const blob=new Blob([records.map(x=>JSON.stringify(x)).join('\\n')+'\\n'],{type:'application/x-ndjson'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reviews.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};</script></html>'''

