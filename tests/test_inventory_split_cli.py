from PIL import Image

from geoguessr_benchmark import __main__
from geoguessr_benchmark import inventory
from geoguessr_benchmark.cleaning import HUD_BOXES


def test_cli_inventory_split_and_verify_fixture(tmp_path, monkeypatch, capsys):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/specs.md").write_text("fixture spec", encoding="utf-8")
    directory = tmp_path / "data/compressed_dataset/A"
    directory.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (8, 8), (i * 50, 20, 60)).save(directory / f"{i}.jpg")
    monkeypatch.setattr(inventory, "DEFAULT_BASELINE", {"images": 4, "classes": 1})
    monkeypatch.setitem(HUD_BOXES, (8, 8), ((1, 1, 2, 2), (4, 4, 6, 6)))
    assert __main__.main(["--root", str(tmp_path), "inventory"]) == 1
    assert __main__.main(["--root", str(tmp_path), "clean"]) == 0
    raw = directory / "0.jpg"
    original = raw.read_bytes()
    raw.write_bytes(original + b"changed bytes")
    assert __main__.main(["--root", str(tmp_path), "inventory"]) == 1
    assert "raw SHA-256 mismatch" in capsys.readouterr().err
    raw.write_bytes(original)
    assert __main__.main(["--root", str(tmp_path), "inventory"]) == 0
    raw.unlink()
    assert __main__.main(["--root", str(tmp_path), "split"]) == 1
    assert "raw" in capsys.readouterr().err
    assert not (tmp_path / "benchmark_lock.json").exists()
    raw.write_bytes(original)
    assert __main__.main(["--root", str(tmp_path), "split"]) == 0
    assert __main__.main(["--root", str(tmp_path), "verify", "--stage", "splits"]) == 0
    assert (tmp_path / "reports/split_audit.json").exists()
    assert (tmp_path / "benchmark_lock.json").exists()
    for split in ("train", "val", "test"):
        manifest = (tmp_path / f"splits/{split}.csv").read_text(encoding="utf-8")
        assert "data/cleaned_dataset/" in manifest and "data/compressed_dataset/" not in manifest
    assert __main__.main(["--root", str(tmp_path), "verify", "--stage", "preflight"]) != 0
    assert "required stage not published" in capsys.readouterr().err

