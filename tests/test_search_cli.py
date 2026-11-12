import json

import pytest

from arm64_rag.cli import main


def run_cli(monkeypatch, capsys, *args):
    monkeypatch.setattr("sys.argv", ["arm64-rag", *map(str, args)])
    main()
    return json.loads(capsys.readouterr().out)


def test_index_search_update_and_sync(monkeypatch, capsys, tmp_path):
    source = tmp_path / "guide.md"
    other = tmp_path / "note.txt"
    source.write_text("# Build\n\nEnable GGML_CPU_KLEIDIAI for this build.\n")
    other.write_text("A separate zebrafish note.\n")
    db = tmp_path / "private" / "index.sqlite"
    indexed = run_cli(
        monkeypatch, capsys, "index", source, other, "--root", tmp_path, "--db", db
    )
    assert indexed["documents"] == 2
    found = run_cli(monkeypatch, capsys, "search", "GGML_CPU_KLEIDIAI", "--db", db)
    assert found["results"][0]["source_id"] == "guide.md"
    assert "GGML_CPU_KLEIDIAI" in found["results"][0]["text"]
    assert found["results"][0]["line_start"] == 1
    source.write_text("# Build\n\nUse a different aardvark flag.\n")
    run_cli(monkeypatch, capsys, "index", source, "--root", tmp_path, "--db", db)
    assert (
        run_cli(monkeypatch, capsys, "search", "GGML_CPU_KLEIDIAI", "--db", db)[
            "results"
        ]
        == []
    )
    assert run_cli(monkeypatch, capsys, "search", "zebrafish", "--db", db)["results"]
    synced = run_cli(
        monkeypatch, capsys, "index", source, "--root", tmp_path, "--db", db, "--sync"
    )
    assert synced["documents"] == 1
    assert (
        run_cli(monkeypatch, capsys, "search", "zebrafish", "--db", db)["results"] == []
    )


def test_failed_import_keeps_index_usable(monkeypatch, capsys, tmp_path):
    source = tmp_path / "valid.txt"
    source.write_text("An unchanged platypus record.")
    db = tmp_path / "index.sqlite"
    first = run_cli(
        monkeypatch, capsys, "index", source, "--root", tmp_path, "--db", db
    )
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff")
    source.write_text("A replacement record.")
    monkeypatch.setattr(
        "sys.argv",
        [
            "arm64-rag",
            "index",
            str(source),
            str(bad),
            "--root",
            str(tmp_path),
            "--db",
            str(db),
            "--sync",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "Error:" in capsys.readouterr().err
    result = run_cli(monkeypatch, capsys, "search", "platypus", "--db", db)
    assert result["generation"] == first["generation"]
    assert result["results"]


def test_search_missing_index_has_plain_error_and_creates_no_file(
    monkeypatch, capsys, tmp_path
):
    db = tmp_path / "missing.sqlite"
    monkeypatch.setattr("sys.argv", ["arm64-rag", "search", "hello", "--db", str(db)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("Error:")
    assert "Traceback" not in output.err
    assert not db.exists()


def test_ambiguous_index_inputs_are_rejected(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["arm64-rag", "index", "some.txt", "--manifest", "manifest.json"]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Provide document paths or --manifest" in capsys.readouterr().err


def test_evaluation_failures_return_report_and_nonzero_exit(monkeypatch, capsys):
    report = {"summary": {"cases_failed": 1}, "cases": [{"failure": "Index is busy."}]}
    monkeypatch.setattr("arm64_rag.cli.evaluate_retrieval", lambda *a, **kw: report)
    monkeypatch.setattr("sys.argv", ["arm64-rag", "evaluate"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == report
