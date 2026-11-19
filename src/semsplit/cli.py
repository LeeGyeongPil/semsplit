"""Command line interface: ``semsplit split | explain | parity``."""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from langchain_core.embeddings import Embeddings

from . import __version__
from .chunker import SemanticChunker
from .embeddings import embedding_from_spec
from .errors import EmbeddingBackendError, InvalidConfigError, LegacyNotInstalledError, SemsplitError
from .i18n import SUPPORTED, resolve_lang, translator
from .parity import ParityReport, check_parity, load_golden, record_golden
from .thresholds import BREAKPOINT_DEFAULTS

EXIT_OK, EXIT_MISMATCH, EXIT_USAGE, EXIT_NO_LEGACY, EXIT_BACKEND = 0, 1, 2, 3, 4
BAR_WIDTH = 24

Translator = Callable[..., str]


def _peek_lang(argv: Sequence[str]) -> Optional[str]:
    """Find ``--lang`` before real parsing so the help text can already be translated."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose error prefix and section titles are translated."""

    t: Translator = staticmethod(translator("en"))  # replaced per run

    def error(self, message: str) -> "Any":
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: {self.t('err.prefix')}: {message}\n")


def _formatter(t: Translator) -> "type[argparse.HelpFormatter]":
    class Formatter(argparse.HelpFormatter):
        def add_usage(self, usage, actions, groups, prefix=None):  # type: ignore[no-untyped-def]
            super().add_usage(usage, actions, groups, f"{t('help.usage')}: " if prefix is None else prefix)

    return Formatter


def _positive_int(t: Translator) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(t("err.bad_int", value=value)) from None
        if number < 1:
            raise argparse.ArgumentTypeError(t("err.bad_int", value=value))
        return number

    return parse


def _number(t: Translator) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(t("err.bad_number", value=value)) from None
        if not math.isfinite(number):  # float() accepts "nan" and "inf"; no threshold can use them
            raise argparse.ArgumentTypeError(t("err.not_finite", value=value))
        return number

    return parse


def _eps(t: Translator) -> Callable[[str], float]:
    as_number = _number(t)

    def parse(value: str) -> float:
        number = as_number(value)
        if number < 0:  # a negative margin would silently report nothing as fragile
            raise argparse.ArgumentTypeError(t("err.bad_eps", value=value))
        return number

    return parse


# Threshold types whose amount is a percentile (numpy rejects values outside 0..100).
PERCENTILE_TYPES = ("percentile", "gradient")


def _check_amount(args: argparse.Namespace, t: Translator) -> None:
    """Reject a percentile ``--amount`` outside 0..100 before numpy raises a bare ValueError.

    The library keeps the original's ValueError on purpose (parity); on the command line a
    typo must be a usage error (exit 2), not a crash that looks like a parity mismatch (exit 1).
    """
    if args.amount is None or args.number_of_chunks is not None:
        return  # no amount given, or number_of_chunks overrides it
    if args.threshold in PERCENTILE_TYPES and not 0 <= args.amount <= 100:
        raise _UsageError(t("err.amount_range", kind=args.threshold, value=f"{args.amount:g}"))


def _choice(t: Translator, key: str, choices: Sequence[str]) -> Callable[[str], str]:
    def parse(value: str) -> str:
        if value not in choices:
            raise argparse.ArgumentTypeError(t(key, value=value, choices=", ".join(choices)))
        return value

    return parse


def build_parser(lang: str) -> argparse.ArgumentParser:
    """Build the argument parser with all texts in ``lang``."""
    t = translator(lang)
    _Parser.t = staticmethod(t)
    fmt = _formatter(t)
    parser = _Parser(
        prog="semsplit", description=t("app.description"), epilog=t("app.epilog"), formatter_class=fmt
    )
    parser._positionals.title = t("help.positional")
    parser._optionals.title = t("help.options")
    parser.add_argument("--version", action="version", version=f"semsplit {__version__}", help=argparse.SUPPRESS)
    lang_arg = _choice(t, "err.bad_lang", SUPPORTED)
    parser.add_argument("--lang", type=lang_arg, metavar="{en,ko}", help=t("arg.lang"))
    sub = parser.add_subparsers(dest="command", metavar="{split,explain,parity}", title=t("help.commands"), parser_class=_Parser)

    def common(p: argparse.ArgumentParser, *, size_options: bool) -> None:
        p._positionals.title = t("help.positional")
        p._optionals.title = t("help.options")
        p.add_argument("--lang", type=lang_arg, metavar="{en,ko}", default=argparse.SUPPRESS, help=t("arg.lang"))
        p.add_argument("--embeddings", default="hash", metavar="SPEC", help=t("arg.embeddings"))
        p.add_argument(
            "--threshold", default="percentile", metavar="TYPE",
            type=_choice(t, "err.bad_threshold", tuple(BREAKPOINT_DEFAULTS)), help=t("arg.threshold"),
        )
        p.add_argument("--amount", type=_number(t), metavar="N", help=t("arg.amount"))
        p.add_argument("--buffer-size", type=_positive_or_zero(t), default=1, metavar="N", help=t("arg.buffer_size"))
        p.add_argument("--number-of-chunks", type=_positive_int(t), metavar="N", help=t("arg.number_of_chunks"))
        p.add_argument("--sentence-split", default="default", metavar="PRESET",
                       type=_choice(t, "err.bad_choice", ("default", "cjk")), help=t("arg.sentence_split"))
        if size_options:
            p.add_argument("--min-chunk-size", type=_positive_int(t), metavar="N", help=t("arg.min_chunk_size"))
            p.add_argument("--max-chunk-size", type=_positive_int(t), metavar="N", help=t("arg.max_chunk_size"))
            p.add_argument("--text-mode", metavar="MODE", default=None,
                           type=_choice(t, "err.bad_choice", ("legacy", "original")), help=t("arg.text_mode"))
            p.add_argument("--embed-batch-size", type=_positive_int(t), metavar="N", help=t("arg.embed_batch_size"))

    split = sub.add_parser("split", help=t("cmd.split"), description=t("cmd.split"), formatter_class=fmt)
    common(split, size_options=True)
    split.add_argument("files", nargs="+", metavar="FILE", help=t("arg.file"))

    explain = sub.add_parser("explain", help=t("cmd.explain"), description=t("cmd.explain"), formatter_class=fmt)
    common(explain, size_options=False)
    explain.add_argument("file", metavar="FILE", help=t("arg.file"))
    explain.add_argument("--eps", type=_eps(t), default=1e-9, metavar="EPS", help=t("arg.eps"))

    parity = sub.add_parser("parity", help=t("cmd.parity"), description=t("cmd.parity"), formatter_class=fmt)
    common(parity, size_options=False)
    parity.add_argument("files", nargs="*", metavar="FILE", help=t("arg.files"))
    parity.add_argument("--against", choices=["legacy"], metavar="legacy", help=t("arg.against"))
    parity.add_argument("--golden", metavar="FILE", help=t("arg.golden"))
    parity.add_argument("--record", metavar="FILE", help=t("arg.record"))
    parity.add_argument("--json", action="store_true", help=t("arg.json"))
    parity.add_argument("--eps", type=_eps(t), default=1e-9, metavar="EPS", help=t("arg.eps"))
    return parser


def _positive_or_zero(t: Translator) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(t("err.bad_nonneg_int", value=value)) from None
        if number < 0:  # 0 is valid here (--buffer-size 0 embeds each sentence alone)
            raise argparse.ArgumentTypeError(t("err.bad_nonneg_int", value=value))
        return number

    return parse


# ---- helpers ----------------------------------------------------------------------------


def _read(path: str, t: Translator) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _UsageError(t("err.file", path=path, reason=exc)) from exc


class _UsageError(Exception):
    pass


def _chunker_options(args: argparse.Namespace) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "breakpoint_threshold_type": args.threshold,
        "breakpoint_threshold_amount": args.amount,
        "buffer_size": args.buffer_size,
        "number_of_chunks": args.number_of_chunks,
        "sentence_split": args.sentence_split,
    }
    for name in ("min_chunk_size", "max_chunk_size", "text_mode", "embed_batch_size"):
        value = getattr(args, name, None)
        if value is not None:
            opts[name] = value
    return opts


def _rule_label(chunker: SemanticChunker) -> str:
    if chunker.number_of_chunks is not None:
        return f"number_of_chunks={chunker.number_of_chunks}"
    return f"{chunker.breakpoint_threshold_type} {chunker.breakpoint_threshold_amount:g}"


def _cells(text: str) -> int:
    """Display width in terminal cells: full-width and wide characters take two."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, *, right: bool = False) -> str:
    """Pad ``text`` with spaces to ``width`` cells (``str.format`` counts characters, not cells)."""
    gap = " " * max(width - _cells(text), 0)
    return gap + text if right else text + gap


def _clip(text: str, width: int) -> str:
    """Cut ``text`` so that it fits in ``width`` cells."""
    out, used = [], 0
    for ch in text:
        w = _cells(ch)
        if used + w > width:
            break
        out.append(ch)
        used += w
    return "".join(out)


def _lib_message(exc: SemsplitError, t: Translator) -> str:
    """Message of a library error in the screen language (English text when it has no key)."""
    return t(exc.key, **exc.params) if exc.key else str(exc)


def _preview(sentence: str, width: int = 38) -> str:
    flat = " ".join(sentence.split())
    return flat if _cells(flat) <= width else _clip(flat, width - 3) + "..."


def _bar(value: float, top: float, threshold: Optional[float]) -> str:
    cells = [" "] * (BAR_WIDTH + 1)
    filled = round(value / top * BAR_WIDTH) if top > 0 else 0
    for i in range(min(max(filled, 0), BAR_WIDTH)):
        cells[i] = "#"
    if threshold is not None and top > 0:
        cells[min(max(round(threshold / top * BAR_WIDTH), 0), BAR_WIDTH)] = "|"
    return "".join(cells)


def _config_label(config: Mapping[str, Any], t: Translator) -> str:
    short = {"breakpoint_threshold_type": "type", "breakpoint_threshold_amount": "amount"}
    return ",".join(f"{short.get(k, k)}={v}" for k, v in config.items()) or t("parity.defaults")


# ---- commands ---------------------------------------------------------------------------


def _cmd_split(args: argparse.Namespace, emb: Embeddings, t: Translator) -> int:
    chunker = SemanticChunker(emb, add_start_index=True, **_chunker_options(args))
    for path in args.files:
        text = _read(path, t)
        if not text.strip():  # the library keeps one empty chunk (parity); on the CLI say so instead
            print(t("split.empty_file", path=path), file=sys.stderr)
            continue
        for n, doc in enumerate(chunker.create_documents([text])):
            row = {"source": path, "chunk": n, "start_index": doc.metadata["start_index"], "text": doc.page_content}
            print(json.dumps(row, ensure_ascii=False))
    return EXIT_OK


def _cmd_explain(args: argparse.Namespace, emb: Embeddings, t: Translator) -> int:
    text = _read(args.file, t)
    chunker = SemanticChunker(emb, **_chunker_options(args))
    info = chunker.analyze(text, eps=args.eps)
    count = len(info.sentences) if text.strip() else 0  # blank input has no sentences to show
    print(t("explain.header", path=args.file))
    print(t("explain.sentences", count=count))
    if info.threshold is None:
        print(t("explain.too_short", count=count))
        return EXIT_OK
    gradient = chunker.number_of_chunks is None and chunker.breakpoint_threshold_type == "gradient"
    print(t("explain.threshold", value=info.threshold, rule=_rule_label(chunker)))
    print(t("explain.legend_gradient" if gradient else "explain.legend"))
    top = max(info.distances) or 1.0
    for i, distance in enumerate(info.distances):
        marks = []
        if i in info.breakpoints:
            marks.append(t("explain.cut"))
        if i in info.fragile:
            marks.append(t("explain.fragile"))
        extra = f" grad={info.values[i]:+.4f}" if gradient else ""
        bar = _bar(distance, top, None if gradient else info.threshold)
        print(f"{i + 1:>3} {bar} {distance:.4f}{extra} {_pad(' '.join(marks), 14)} {_preview(info.sentences[i])}".rstrip())
    print(t("explain.summary", sentences=len(info.sentences), cuts=len(info.breakpoints),
            chunks=len(info.breakpoints) + 1, fragile=len(info.fragile)))
    return EXIT_OK


def _doc_names(paths: Sequence[str]) -> dict[str, str]:
    """Display name per path: the file name, or the full path when names collide."""
    names = [Path(p).name for p in paths]
    return {p: (n if names.count(n) == 1 else p) for p, n in zip(paths, names)}


def _row(doc: str, config: str, result: str, chunks: str, fragile: str) -> str:
    """One table row; every cell is padded by display width so Korean text stays aligned."""
    cells = [_pad(doc, 22), _pad(config, 44), _pad(result, 9), _pad(chunks, 6, right=True), _pad(fragile, 7, right=True)]
    return " ".join(cells).rstrip()


def _print_report(report: ParityReport, embeddings_spec: str, t: Translator) -> None:
    mock = " (mock)" if embeddings_spec == "hash" else ""
    print(t("parity.header", against=report.against, embeddings=f"{embeddings_spec}{mock}"))
    print(_row(t("parity.col_doc"), t("parity.col_config"), t("parity.col_result"),
               t("parity.col_chunks"), t("parity.col_fragile")))
    for case in report.cases:
        result = t("parity.equal") if case.equal else t("parity.differs")
        print(_row(_clip(case.doc, 22), _clip(_config_label(case.config, t), 44), result,
                   str(len(case.new_chunks)), str(len(case.fragile))))
        if not case.equal:
            d = case.diff_boundaries
            print(t("parity.detail_boundaries", old=d.get("only_legacy", []), new=d.get("only_new", [])))
            if case.first_mismatch:
                print(t("parity.detail_first", chunk=case.first_mismatch["chunk_index"],
                        offset=case.first_mismatch["char_offset"]))
    bad = sum(not c.equal for c in report.cases)
    if bad:
        print(t("parity.summary_fail", bad=bad, count=len(report.cases)))
    else:
        print(t("parity.summary_ok", count=len(report.cases), fragile=sum(len(c.fragile) for c in report.cases)))


def _cmd_parity(args: argparse.Namespace, emb: Embeddings, t: Translator) -> int:
    if args.golden and (args.against or args.record):
        raise _UsageError(t("err.conflict"))
    if args.golden:
        golden = load_golden(args.golden)
        spec = args.embeddings if args.embeddings != "hash" else str(golden.get("embeddings", "hash"))
        report = check_parity([], embedding_from_spec(spec) if spec != args.embeddings else emb,
                              against=args.golden, eps=args.eps)
    else:
        if not args.files:
            raise _UsageError(t("err.no_input"))
        names = _doc_names(args.files)
        texts = {names[p]: _read(p, t) for p in args.files}
        spec = args.embeddings
        config = {k: v for k, v in {
            "breakpoint_threshold_type": args.threshold if args.threshold != "percentile" else None,
            "breakpoint_threshold_amount": args.amount,
            "buffer_size": args.buffer_size if args.buffer_size != 1 else None,
            "number_of_chunks": args.number_of_chunks,
        }.items() if v is not None}
        configs = [config] if config else None  # explicit flags narrow the run to that config
        if args.record:
            source = "legacy" if args.against else "semsplit"
            data = record_golden(texts, args.record, emb, configs, embeddings_name=spec, source=source)
            print(t("parity.recorded", count=len(data["cases"]), path=args.record))
            return EXIT_OK
        report = check_parity(texts, emb, configs, against="legacy", eps=args.eps)
    if args.json:
        payload = report.to_dict()
        payload.update({"embeddings": spec, "mock": spec == "hash"})
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_report(report, spec, t)
    return EXIT_OK if report.all_equal else EXIT_MISMATCH


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    lang_value = _peek_lang(args_list)
    lang = resolve_lang(lang_value if lang_value in SUPPORTED else None)
    t = translator(lang)
    parser = build_parser(lang)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args = parser.parse_args(args_list)
    if args.command is None:
        parser.print_help()
        return EXIT_USAGE
    # --lang is also validated by argparse; resolve again with the parsed value.
    lang = resolve_lang(args.lang)
    t = translator(lang)
    try:
        _check_amount(args, t)  # 범위 밖 백분위는 numpy 오류(종료 1) 대신 사용법 오류(종료 2)로
        emb = embedding_from_spec(args.embeddings)
        if args.embeddings == "hash" and not getattr(args, "json", False):
            print(t("mock.note"), file=sys.stderr)
        handler = {"split": _cmd_split, "explain": _cmd_explain, "parity": _cmd_parity}[args.command]
        return handler(args, emb, t)
    except _UsageError as exc:
        print(f"semsplit: {t('err.prefix')}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except LegacyNotInstalledError:
        print(f"semsplit: {t('err.prefix')}: {t('err.legacy_missing')}", file=sys.stderr)
        return EXIT_NO_LEGACY
    except EmbeddingBackendError as exc:
        message = _lib_message(exc, t) if exc.key else t("err.backend", message=exc)
        print(f"semsplit: {t('err.prefix')}: {message}", file=sys.stderr)
        return EXIT_BACKEND
    except (InvalidConfigError, SemsplitError) as exc:
        print(f"semsplit: {t('err.prefix')}: {_lib_message(exc, t)}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
