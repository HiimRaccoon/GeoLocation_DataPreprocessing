import json

import pytest

from test_qc import setup_fixture
from geoguessr_benchmark.augmentation.review import export_review_queue, import_review_decisions


def review_candidate(root, candidate, letter):
    import shutil
    sid = letter * 64
    relative = f"data/augmented_dataset/da_fusion/{candidate['class_name']}/{sid}.jpg"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / candidate["synthetic_path"], target)
    candidate.update({"synthetic_id":sid, "synthetic_path":relative, "attempt_index":0,
                      "augmentation_type":"da_fusion", "filter_status":"pending", "qc_passed":True})


def test_review_bundle_roundtrip_checks_image_hashes(tmp_path):
    source, candidate = setup_fixture(tmp_path)
    review_candidate(tmp_path, candidate, "a")
    manifests = tmp_path / "augmentation/manifests"
    manifests.mkdir(parents=True)
    (manifests / "candidates.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    queue = export_review_queue(tmp_path, "augmentation/manifests/review_queue.jsonl")
    assert len(queue) == 1
    bundle = tmp_path / "augmentation/review"
    assert (bundle / "index.html").exists()
    assert (bundle / "sources" / (("a"*64) + ".jpg")).read_bytes() == (tmp_path / source["filepath"]).read_bytes()
    decision = {"synthetic_id":"a"*64, "source_sha256":candidate["source_sha256"],
                "output_sha256":candidate["output_sha256"], "reviewer_id":"reviewer",
                "review_timestamp":"2026-11-14T00:00:00Z", "rubric_version":queue[0]["rubric_version"],
                "decision":"accept", "reason":None}
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(decision) + "\n", encoding="utf-8")
    assert len(import_review_decisions(tmp_path, decisions)) == 1
    from geoguessr_benchmark.__main__ import main
    assert main(["--root", str(tmp_path), "review-import", "--input", str(decisions)]) == 0
    (tmp_path / candidate["synthetic_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        import_review_decisions(tmp_path, decisions)


def test_offline_html_exports_usable_jsonl(tmp_path):
    import shutil
    import subprocess
    if shutil.which("node") is None:
        pytest.skip("Node.js is needed to execute the offline review JavaScript")
    source, candidate = setup_fixture(tmp_path)
    review_candidate(tmp_path, candidate, "b")
    manifests = tmp_path / "augmentation/manifests"
    manifests.mkdir(parents=True)
    (manifests / "candidates.jsonl").write_text(json.dumps(candidate)+"\n", encoding="utf-8")
    export_review_queue(tmp_path, "augmentation/manifests/review_queue.jsonl")
    html = (tmp_path / "augmentation/review/index.html").read_text(encoding="utf-8")
    script = html.split("<script>",1)[1].split("</script>",1)[0]
    harness = r'''const nodes = {};
class Element {
  constructor(tag) {this.tag=tag;this.children=[];this.value='';this.textContent='';}
  append(x){this.children.push(x)}
  click(){if(this.onclick)this.onclick();}
}
global.document={getElementById(id){return nodes[id] ||= new Element(id)},createElement(tag){return new Element(tag)}};
global.URL={createObjectURL(blob){global.download=blob;return 'blob:test'},revokeObjectURL(){}};
global.alert=(x)=>{throw new Error(x)};
global.setTimeout=()=>{};
'''
    run = harness + script + r'''
document.getElementById('reviewer').value='reviewer';
const article=nodes.items.children[0];
article.children.find(x=>x.tag==='button' && x.textContent==='Reject — ambiguous').click();
nodes.save.click();
global.download.text().then(t=>process.stdout.write(t));
'''
    execution = subprocess.run(["node", "-e", run], capture_output=True, text=True, encoding="utf-8")
    assert execution.returncode == 0, execution.stderr
    decision = json.loads(execution.stdout)
    assert decision["reason"] == "reject_ambiguous" and decision["decision"] == "reject"
    path = tmp_path / "reviews.jsonl"
    path.write_text(execution.stdout, encoding="utf-8")
    assert import_review_decisions(tmp_path, path)[0]["synthetic_id"] == "b" * 64




def test_import_rejects_conflicts_and_does_not_write_partial_reviews(tmp_path):
    source, candidate = setup_fixture(tmp_path)
    review_candidate(tmp_path, candidate, "c")
    manifests = tmp_path / "augmentation/manifests"
    manifests.mkdir(parents=True)
    (manifests / "candidates.jsonl").write_text(json.dumps(candidate)+"\n", encoding="utf-8")
    queue = export_review_queue(tmp_path, "augmentation/manifests/review_queue.jsonl")
    base = {"synthetic_id":"c"*64, "source_sha256":candidate["source_sha256"], "output_sha256":candidate["output_sha256"],
            "reviewer_id":"r", "review_timestamp":"2026-11-14T00:00:00Z", "rubric_version":queue[0]["rubric_version"],
            "decision":"accept", "reason":None}
    invalid = dict(base, reason="reject_ambiguous")
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(base)+"\n"+json.dumps(invalid)+"\n", encoding="utf-8")
    with pytest.raises(ValueError):
        import_review_decisions(tmp_path, path)
    assert not (manifests / "reviews.jsonl").exists()
    path.write_text(json.dumps(invalid)+"\n", encoding="utf-8")
    with pytest.raises(ValueError):
        import_review_decisions(tmp_path, path)
    assert not (manifests / "reviews.jsonl").exists()


def test_review_import_rejects_missing_input(tmp_path):
    setup_fixture(tmp_path)
    with pytest.raises((ValueError, OSError)):
        import_review_decisions(tmp_path, tmp_path / "missing.jsonl")


def test_review_bundle_works_from_copied_file_url_in_browser(tmp_path):
    import shutil
    from pathlib import Path
    playwright = pytest.importorskip("playwright.sync_api")
    browsers = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"), Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]
    executable = next((path for path in browsers if path.is_file()),None)
    if executable is None:
        pytest.skip("local Chromium browser unavailable")
    source,candidate=setup_fixture(tmp_path)
    review_candidate(tmp_path, candidate, "d")
    manifests=tmp_path/"augmentation/manifests"
    manifests.mkdir(parents=True)
    (manifests/"candidates.jsonl").write_text(json.dumps(candidate)+"\n",encoding="utf-8")
    export_review_queue(tmp_path,"augmentation/manifests/review_queue.jsonl")
    copied=tmp_path/"copied-review"
    shutil.copytree(tmp_path/"augmentation/review",copied)
    with playwright.sync_playwright() as runtime:
        browser=runtime.chromium.launch(executable_path=str(executable),headless=True,args=["--disable-background-networking","--proxy-server=http://127.0.0.1:9"])
        context=browser.new_context(accept_downloads=True)
        page=context.new_page()
        page.route("http://**",lambda route:route.abort())
        page.route("https://**",lambda route:route.abort())
        page.goto((copied/"index.html").as_uri())
        assert page.locator("img").count()==2
        assert page.locator("img").evaluate_all("images => images.every(image => image.complete && image.naturalWidth > 0)")
        page.locator("#reviewer").fill("offline-reviewer")
        page.get_by_role("button",name="Accept",exact=True).click()
        with page.expect_download() as download_info:
            page.get_by_role("button",name="Download reviews.jsonl").click()
        result=tmp_path/"browser-reviews.jsonl"
        download_info.value.save_as(result)
        browser.close()
    decision=json.loads(result.read_text(encoding="utf-8"))
    assert decision["reviewer_id"]=="offline-reviewer" and decision["decision"]=="accept"
    assert import_review_decisions(tmp_path,result)[0]["synthetic_id"]=="d"*64


def test_review_export_never_copies_holdout_as_generated_image(tmp_path):
    import csv
    source,candidate=setup_fixture(tmp_path)
    with (tmp_path/"splits/test.csv").open(encoding="utf-8",newline="") as file:
        holdout=next(csv.DictReader(file))
    candidate.update({"synthetic_id":"e"*64,"attempt_index":0,"augmentation_type":"da_fusion","filter_status":"pending","qc_passed":True,
                      "synthetic_path":holdout["filepath"],"output_sha256":holdout["sha256"]})
    manifests=tmp_path/"augmentation/manifests"
    manifests.mkdir(parents=True)
    (manifests/"candidates.jsonl").write_text(json.dumps(candidate)+"\n",encoding="utf-8")
    with pytest.raises(ValueError):
        export_review_queue(tmp_path,"augmentation/manifests/review_queue.jsonl")
    assert not (tmp_path/"augmentation/review/generated"/("e"*64+".jpg")).exists()


def test_review_export_rechecks_automatic_qc_at_preflight(tmp_path,monkeypatch):
    from geoguessr_benchmark.artifacts import _REQUIRED,atomic_write_bytes,canonical_json_bytes,sha256_file
    import geoguessr_benchmark.config as config_module
    source,candidate=setup_fixture(tmp_path)
    review_candidate(tmp_path,candidate,"f")
    manifests=tmp_path/"augmentation/manifests"
    manifests.mkdir(parents=True)
    (manifests/"candidates.jsonl").write_text(json.dumps(candidate)+"\n",encoding="utf-8")
    lock=json.loads((tmp_path/"benchmark_lock.json").read_text(encoding="utf-8"))
    for name in _REQUIRED["preflight"]:
        if not (tmp_path/name).exists(): atomic_write_bytes(tmp_path/name,b"fixture")
        lock["artifacts"][name]=sha256_file(tmp_path/name)
    lock["stage"]="preflight"
    atomic_write_bytes(tmp_path/"benchmark_lock.json",canonical_json_bytes(lock))
    monkeypatch.setattr(config_module,"load_config",lambda root:{"da_fusion":{}})
    with pytest.raises(ValueError,match="QC"):
        export_review_queue(tmp_path,"augmentation/manifests/review_queue.jsonl")
