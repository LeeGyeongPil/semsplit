"""AC-3: no legacy modules, no deprecation warnings, two runtime dependencies."""

import re
import subprocess
import sys
import warnings
from importlib.metadata import requires

from helpers import ROOT, data
from semsplit import HashEmbeddings, SemanticChunker

SCRIPT = """
import sys
import semsplit
text = open(sys.argv[1], encoding="utf-8").read()
c = semsplit.SemanticChunker(semsplit.HashEmbeddings(), breakpoint_threshold_amount=50, add_start_index=True)
assert len(c.split_text(text)) > 1
c.create_documents([text])
bad = [m for m in sys.modules if m.split(".")[0] in ("langchain_community", "langchain_experimental")]
assert not bad, bad
print("clean")
"""


def test_no_legacy_modules_loaded():
    # A fresh interpreter, so modules loaded by other tests cannot hide a problem.
    out = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT, str(ROOT / "examples" / "data" / "three_topics.en.txt")],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "clean"


def test_no_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chunker = SemanticChunker(HashEmbeddings(), breakpoint_threshold_amount=50)
        chunker.split_text(data("three_topics.en.txt"))
    assert [w for w in caught if issubclass(w.category, DeprecationWarning) and "semsplit" in str(w.filename)] == []
    assert not [w for w in caught if "sunset" in str(w.message)]


def test_runtime_dependencies():
    names = set()
    for req in requires("semsplit") or []:
        if "extra ==" in req:
            continue
        names.add(re.split(r"[\s<>=!~;\[(]", req, maxsplit=1)[0].lower().replace("_", "-"))
    assert names == {"langchain-core", "numpy"}
