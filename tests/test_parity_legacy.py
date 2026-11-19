"""AC-2: live comparison with the archived original (skipped when it is not installed)."""

import pytest

from helpers import data
from semsplit import LegacyNotInstalledError, check_parity

pytestmark = pytest.mark.legacy


def test_live_parity_with_same_vectors():
    pytest.importorskip("langchain_experimental.text_splitter")
    docs = {"en": data("three_topics.en.txt"), "ko": data("ko.txt"), "ml": data("multiline.en.txt")}
    report = check_parity(docs, against="legacy")
    assert report.all_equal
    assert len(report.cases) == 3 * 8
    assert all("fragile" in c.to_dict() if hasattr(c, "to_dict") else True for c in report.cases)


def test_missing_legacy_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "langchain_experimental", None)
    monkeypatch.setitem(__import__("sys").modules, "langchain_experimental.text_splitter", None)
    with pytest.raises(LegacyNotInstalledError, match=r"pip install 'semsplit\[legacy\]'"):
        check_parity({"en": "A. B."}, against="legacy")
