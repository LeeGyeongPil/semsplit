"""AC-9: English and Korean screens, same keys, documented precedence."""

import string

import pytest

from helpers import ROOT
from semsplit.cli import main
from semsplit.i18n import load_messages, resolve_lang, translator

EN = str(ROOT / "examples" / "data" / "three_topics.en.txt")
HANGUL = lambda s: any("가" <= ch <= "힣" for ch in s)  # noqa: E731


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("SEMSPLIT_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(name, raising=False)


def test_both_languages_have_exactly_the_same_keys():
    assert set(load_messages("en")) == set(load_messages("ko"))


def test_placeholders_match_between_languages():
    fields = lambda s: {f for _, f, _, _ in string.Formatter().parse(s) if f}  # noqa: E731
    en, ko = load_messages("en"), load_messages("ko")
    for key in en:
        assert fields(en[key]) == fields(ko[key]), key


def test_english_table_has_no_hangul_and_korean_table_has():
    assert not any(HANGUL(v) for v in load_messages("en").values())
    ko = load_messages("ko")
    plain = [k for k, v in ko.items() if not HANGUL(v)]
    assert set(plain) <= {"parity.columns", "explain.header", "mock.note"}, plain


def test_precedence_flag_over_env_var_over_locale():
    env = {"SEMSPLIT_LANG": "ko", "LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}
    assert resolve_lang("en", env) == "en"  # --lang wins
    assert resolve_lang(None, env) == "ko"  # then SEMSPLIT_LANG
    assert resolve_lang(None, {"LC_ALL": "ko_KR.UTF-8", "LANG": "en_US.UTF-8"}) == "ko"
    assert resolve_lang(None, {"LC_ALL": "en_US.UTF-8", "LANG": "ko_KR.UTF-8"}) == "en"  # LC_ALL beats LANG
    assert resolve_lang(None, {"LC_MESSAGES": "ko_KR.UTF-8", "LANG": "en_US"}) == "ko"
    assert resolve_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"


@pytest.mark.parametrize("env", [{}, {"LANG": "C"}, {"LANG": "fr_FR.UTF-8"}, {"LC_ALL": "ja_JP.UTF-8"}, {"SEMSPLIT_LANG": "fr"}])
def test_anything_but_korean_is_english(env):
    assert resolve_lang(None, env) == "en"


def test_cli_language_switch_changes_output(capsys):
    main(["explain", EN, "--amount", "50", "--lang", "en"])
    en_out = capsys.readouterr()
    main(["explain", EN, "--amount", "50", "--lang", "ko"])
    ko_out = capsys.readouterr()
    assert "threshold:" in en_out.out and "summary:" in en_out.out and not HANGUL(en_out.out + en_out.err)
    assert "임계값" in ko_out.out and "요약:" in ko_out.out and "연습용 목" in ko_out.err


def test_help_is_translated(capsys):
    outputs = {}
    for lang in ("en", "ko"):
        with pytest.raises(SystemExit) as exc:
            main(["explain", "--help", "--lang", lang])
        assert exc.value.code == 0
        outputs[lang] = capsys.readouterr().out
    assert "usage:" in outputs["en"] and "distance bars" in outputs["en"] and not HANGUL(outputs["en"])
    assert "사용법:" in outputs["ko"] and "거리 막대" in outputs["ko"]


def test_env_var_and_locale_choose_cli_language(monkeypatch, capsys):
    monkeypatch.setenv("LANG", "ko_KR.UTF-8")
    main(["explain", EN])
    assert "요약:" in capsys.readouterr().out
    monkeypatch.setenv("SEMSPLIT_LANG", "en")
    main(["explain", EN])
    assert "summary:" in capsys.readouterr().out
    main(["explain", EN, "--lang", "ko"])
    assert "요약:" in capsys.readouterr().out


def test_error_messages_are_translated(capsys):
    with pytest.raises(SystemExit):
        main(["explain", EN, "--threshold", "median", "--lang", "ko"])
    err = capsys.readouterr().err
    assert "오류" in err and "median" in err
    with pytest.raises(SystemExit):
        main(["explain", EN, "--threshold", "median", "--lang", "en"])
    assert "error" in capsys.readouterr().err


def test_translator_falls_back_to_key():
    assert translator("en")("no.such.key") == "no.such.key"


KO = str(ROOT / "examples" / "data" / "ko.txt")


@pytest.mark.parametrize(
    "argv, env",
    [
        (["explain", KO, "--embeddings", "foo"], {}),
        (["split", KO, "--min-chunk-size", "50", "--max-chunk-size", "10"], {}),
        (["parity", "--golden", "missing.json"], {}),
        (["explain", KO, "--embeddings", "ollama:"], {}),
        (["explain", KO, "--embeddings", "ollama:nomic"], {"OLLAMA_HOST": "http://127.0.0.1:9"}),
    ],
)
def test_library_errors_are_korean_with_lang_ko(argv, env, capsys, monkeypatch):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert main(["--lang", "ko", *argv]) != 0
    err = capsys.readouterr().err
    message = err.split("오류:", 1)[1]
    assert HANGUL(message)
    # the English library sentence must not leak through
    assert not any(w in message for w in ("Unknown", "must not", "Cannot read", "Could not", "is empty"))


def test_library_errors_stay_english_by_default(capsys):
    assert main(["explain", KO, "--embeddings", "foo"]) == 2
    assert "Unknown embeddings spec 'foo'" in capsys.readouterr().err


def test_library_error_keeps_english_str_and_key():
    from semsplit import SemanticChunker
    from semsplit.errors import InvalidConfigError

    with pytest.raises(InvalidConfigError) as info:
        SemanticChunker(None, text_mode="x")  # type: ignore[arg-type]
    assert str(info.value).startswith("text_mode must be")
    assert info.value.key == "err.lib.text_mode"


def test_table_alignment_uses_display_width():
    from semsplit.cli import _cells, _pad

    assert _cells("흔들림") == 6 and _cells("abc") == 3
    assert _cells(_pad("흔들림", 10)) == 10 and _cells(_pad("ab", 10, right=True)) == 10


def test_empty_file_gets_a_note_on_the_cli(tmp_path, capsys):
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n")
    assert main(["split", str(empty)]) == 0
    out = capsys.readouterr()
    assert out.out == "" and "empty" in out.err
    assert main(["explain", str(empty)]) == 0
    assert "sentences: 0" in capsys.readouterr().out
