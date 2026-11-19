"""AC-8: CLI behaviour and exit codes."""

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from helpers import ROOT, data
from semsplit.cli import main
from semsplit.embeddings import HashEmbeddings

EN = str(ROOT / "examples" / "data" / "three_topics.en.txt")
KO = str(ROOT / "examples" / "data" / "ko.txt")
GOLDEN = str(ROOT / "tests" / "golden" / "semantic_chunker_0.4.2.json")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("SEMSPLIT_LANG", "LC_ALL", "LC_MESSAGES", "LANG", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_explain_shows_bars_threshold_cuts_and_summary(capsys):
    code, out, err = run(capsys, "explain", EN, "--amount", "50", "--lang", "en")
    assert code == 0
    assert re.search(r"^threshold: \d\.\d{4}", out, re.M)
    cuts = out.count("<- cut")
    assert cuts >= 1
    assert "#" in out and "|" in out
    summary = re.search(r"(\d+) cuts, (\d+) chunks", out)
    assert int(summary.group(2)) == int(summary.group(1)) + 1 == cuts + 1
    assert "practice mock" in err  # honest about the mock embeddings


def test_default_95_cuts_less_than_50(capsys):
    _, out95, _ = run(capsys, "explain", EN, "--lang", "en")
    _, out50, _ = run(capsys, "explain", EN, "--amount", "50", "--lang", "en")
    assert out95.count("<- cut") < out50.count("<- cut")


def test_explain_marks_fragile_boundaries(capsys):
    # With an even number of distances the 50th percentile sits on no value; a huge eps makes all fragile.
    code, out, _ = run(capsys, "explain", EN, "--amount", "50", "--eps", "1", "--lang", "en")
    assert code == 0 and "(fragile)" in out


def test_explain_too_short_document(tmp_path, capsys):
    f = tmp_path / "one.txt"
    f.write_text("Only one sentence.")
    code, out, _ = run(capsys, "explain", str(f), "--lang", "en")
    assert code == 0 and "not enough sentences" in out


def test_split_prints_jsonl_with_exact_start_index(capsys):
    code, out, _ = run(capsys, "split", EN, "--amount", "50", "--text-mode", "original", "--lang", "en")
    assert code == 0
    rows = [json.loads(line) for line in out.splitlines()]
    text = data("three_topics.en.txt")
    assert len(rows) == 5
    assert all(text[r["start_index"] : r["start_index"] + len(r["text"])] == r["text"] for r in rows)


def test_parity_golden_ok_exit_0(capsys):
    code, out, _ = run(capsys, "parity", "--golden", GOLDEN, "--lang", "en")
    assert code == 0 and "all 29 cases are identical" in out


def test_parity_golden_one_changed_character_exit_1(tmp_path, capsys):
    golden = json.loads(open(GOLDEN, encoding="utf-8").read())
    case = next(c for c in golden["cases"] if c["id"] == "en-percentile-50")
    case["chunks"][0] = "X" + case["chunks"][0][1:]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(golden), encoding="utf-8")
    code, out, _ = run(capsys, "parity", "--golden", str(path), "--lang", "en")
    assert code == 1
    assert "DIFFERENT" in out and "first difference: chunk 0, character 0" in out


def test_parity_golden_moved_boundary_prints_positions(tmp_path, capsys):
    golden = json.loads(open(GOLDEN, encoding="utf-8").read())
    case = next(c for c in golden["cases"] if c["id"] == "en-percentile-50")
    case["chunks"][:2] = [" ".join(case["chunks"][:2])]  # remove the first boundary
    case["start_index"] = [0] + case["start_index"][2:]
    path = tmp_path / "moved.json"
    path.write_text(json.dumps(golden), encoding="utf-8")
    code, out, _ = run(capsys, "parity", "--golden", str(path), "--lang", "en")
    assert code == 1
    assert re.search(r"only in the original: \[\d+\]|only in semsplit: \[\d+", out)


def test_parity_json_reports_fragile_per_case(capsys):
    pytest.importorskip("langchain_experimental.text_splitter")
    code, out, _ = run(capsys, "parity", EN, KO, "--against", "legacy", "--embeddings", "hash", "--json")
    assert code == 0
    assert '"all_equal": true' in out
    report = json.loads(out)
    assert len(report["cases"]) == 16
    assert all("fragile" in c for c in report["cases"])
    assert report["mock"] is True


def test_parity_legacy_missing_exit_3(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "langchain_experimental", None)
    monkeypatch.setitem(sys.modules, "langchain_experimental.text_splitter", None)
    code, out, err = run(capsys, "parity", EN, "--against", "legacy", "--lang", "en")
    assert code == 3
    assert "pip install 'semsplit[legacy]'" in err


def test_record_then_compare_roundtrip(tmp_path, capsys):
    path = str(tmp_path / "mine.json")
    code, out, _ = run(capsys, "parity", EN, KO, "--record", path, "--lang", "en")
    assert code == 0 and "recorded 16 cases" in out
    code, out, _ = run(capsys, "parity", "--golden", path, "--lang", "en")
    assert code == 0 and "all 16 cases are identical" in out


def test_invalid_threshold_exit_2(capsys):
    code = None
    with pytest.raises(SystemExit) as exc:
        main(["explain", EN, "--threshold", "median", "--lang", "en"])
    assert exc.value.code == 2
    assert "median" in capsys.readouterr().err


def test_bad_numbers_and_files_exit_2(capsys, tmp_path):
    for argv in (
        ["explain", EN, "--amount", "lots"],
        ["split", EN, "--max-chunk-size", "0"],
        ["split", EN, "--lang", "fr"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
    capsys.readouterr()
    code, _, err = run(capsys, "explain", str(tmp_path / "missing.txt"), "--lang", "en")
    assert code == 2 and "cannot read" in err
    code, _, err = run(capsys, "parity", "--lang", "en")
    assert code == 2 and "at least one text file" in err
    code, _, err = run(capsys, "parity", EN, "--golden", GOLDEN, "--against", "legacy", "--lang", "en")
    assert code == 2


def test_invalid_config_from_library_exit_2(capsys):
    code, _, err = run(capsys, "split", EN, "--min-chunk-size", "500", "--max-chunk-size", "100", "--lang", "en")
    assert code == 2 and "max_chunk_size" in err


def test_no_command_prints_help_exit_2(capsys):
    code, out, _ = run(capsys, "--lang", "en")
    assert code == 2 and "split" in out


class _OllamaHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        vectors = HashEmbeddings().embed_documents(body["input"])
        payload = json.dumps({"embeddings": vectors}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_ollama_backend_roundtrip_and_failure(monkeypatch, capsys):
    try:
        server = HTTPServer(("127.0.0.1", 0), _OllamaHandler)
    except OSError:
        pytest.skip("cannot bind a local port in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{server.server_address[1]}")
        code, out, _ = run(capsys, "explain", EN, "--embeddings", "ollama:fake", "--amount", "50", "--lang", "en")
        assert code == 0 and "summary: 9 sentences, 4 cuts" in out  # same vectors as the hash mock
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    code, _, err = run(capsys, "explain", EN, "--embeddings", "ollama:fake", "--lang", "en")
    assert code == 4 and "ollama serve" in err


def test_unknown_embeddings_spec_exit_2(capsys):
    code, _, err = run(capsys, "explain", EN, "--embeddings", "nope", "--lang", "en")
    assert code == 2 and "Unknown embeddings spec" in err


# ---- out-of-range numbers are usage errors (exit 2), never a traceback or exit 1 -------------


def run_any(capsys, *argv):
    """Like ``run`` but also accepts argparse's ``SystemExit`` for options it rejects itself."""
    try:
        code = main(list(argv))
    except SystemExit as exc:
        code = exc.code
    out = capsys.readouterr()
    return code, out.out, out.err


COMMANDS = {
    "split": ["split", KO],
    "explain": ["explain", KO],
    "parity": ["parity", KO, "--against", "legacy"],  # checked before the legacy import
}


@pytest.mark.parametrize("command", sorted(COMMANDS))
@pytest.mark.parametrize("amount", ["-5", "150", "nan", "inf"])
@pytest.mark.parametrize("lang", ["en", "ko"])
def test_out_of_range_percentile_amount_is_a_usage_error(capsys, command, amount, lang):
    code, out, err = run_any(capsys, *COMMANDS[command], "--amount", amount, "--lang", lang)
    assert code == 2
    assert "Traceback" not in err and "ValueError" not in err
    assert out == ""
    if lang == "ko":
        assert re.search(r"오류: .*[가-힣]", err)
        assert "between 0 and 100" not in err
    else:
        assert "error: " in err and not re.search(r"[가-힣]", err)
    if amount in ("-5", "150"):
        assert ("between 0 and 100" if lang == "en" else "0~100") in err
    else:
        assert ("finite number" if lang == "en" else "유한한 숫자") in err


def test_gradient_amount_is_checked_too(capsys):
    code, _, err = run_any(capsys, "explain", KO, "--threshold", "gradient", "--amount", "101", "--lang", "en")
    assert code == 2 and "--threshold gradient" in err and "got 101" in err


def test_amount_limits_only_apply_to_percentile_types(capsys):
    # standard_deviation and interquartile take any finite factor; 0 and 100 are valid percentiles
    for argv in (
        ["explain", KO, "--threshold", "standard_deviation", "--amount", "-5"],
        ["explain", KO, "--threshold", "interquartile", "--amount", "150"],
        ["explain", KO, "--amount", "0"],
        ["explain", KO, "--amount", "100"],
        ["explain", KO, "--number-of-chunks", "3", "--amount", "150"],  # amount is unused here
    ):
        code, _, err = run_any(capsys, *argv, "--lang", "en")
        assert code == 0, (argv, err)


def test_eps_must_be_finite_and_not_negative(capsys):
    for value, words in (("-1", "0 or greater"), ("nan", "finite number"), ("inf", "finite number")):
        code, _, err = run_any(capsys, "explain", KO, "--eps", value, "--lang", "en")
        assert code == 2 and words in err


@pytest.mark.parametrize("lang, words", [("en", "0 or a positive integer"), ("ko", "0 이상의 정수")])
def test_buffer_size_message_says_zero_is_allowed(capsys, lang, words):
    code, _, err = run_any(capsys, "split", KO, "--buffer-size", "-1", "--lang", lang)
    assert code == 2 and words in err
    assert "not a positive integer" not in err and "양의 정수가 아닙니다" not in err
    code, out, _ = run_any(capsys, "split", KO, "--buffer-size", "0", "--lang", lang)
    assert code == 0 and out.count('"chunk"') >= 1
