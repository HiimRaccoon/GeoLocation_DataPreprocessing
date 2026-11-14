from pathlib import Path

from PIL import Image

from geoguessr_benchmark import __main__
from geoguessr_benchmark import inventory


def test_cli_inventory_split_and_verify_fixture(tmp_path, monkeypatch, capsys):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/specs.md").write_text("fixture spec", encoding="utf-8")
    directory = tmp_path / "data/compressed_dataset/A"
    directory.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (8, 8), (i * 50, 20, 60)).save(directory / f"{i}.jpg")
    monkeypatch.setattr(inventory, "DEFAULT_BASELINE", {"images": 4, "classes": 1})
    assert __main__.main(["--root", str(tmp_path), "inventory"]) == 0
    assert __main__.main(["--root", str(tmp_path), "split"]) == 0
    assert __main__.main(["--root", str(tmp_path), "verify", "--stage", "splits"]) == 0
    assert (tmp_path / "reports/split_audit.json").exists()
    assert (tmp_path / "benchmark_lock.json").exists()
    assert __main__.main(["--root", str(tmp_path), "verify", "--stage", "preflight"]) != 0
    assert "required stage not published" in capsys.readouterr().err

