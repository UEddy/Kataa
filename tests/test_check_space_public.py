"""Link classification for the logged-out Space check. Runs in: cloud session (no network)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "csp", Path(__file__).parents[1] / "scripts" / "check_space_public.py")
csp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(csp)


def test_hf_links_are_strict_and_others_manual():
    text = """
    <a href="https://huggingface.co/datasets/JustEddy1/kataa-bench">data</a>
    <a href='https://hf.co/JustEddy1/kataa'>model</a>
    fetch("https://justeddy1-kataa-demo.hf.space/data/heldout.json")
    see https://lexsi.ai/hackathon, https://docs.cohere.com/docs/tiny-aya.
    [paper](https://arxiv.org/abs/2305.18290) and https://github.com/coinmetrics/data
    `https://huggingface.co/${repo}` is a template, not a link
    """
    hf, manual = csp.classify(csp.extract_links(text))
    assert hf == ["https://hf.co/JustEddy1/kataa",
                  "https://huggingface.co/datasets/JustEddy1/kataa-bench",
                  "https://justeddy1-kataa-demo.hf.space/data/heldout.json"]
    assert manual == ["https://arxiv.org/abs/2305.18290", "https://docs.cohere.com/docs/tiny-aya",
                      "https://github.com/coinmetrics/data", "https://lexsi.ai/hackathon"]


def test_lookalike_hosts_are_not_trusted_as_hf():
    assert not csp.is_hf("https://huggingface.co.evil.example/x")
    assert not csp.is_hf("https://nothf.space/x")
    assert csp.is_hf("https://cdn-lfs.huggingface.co/x")


def test_space_host_comes_from_the_hub_metadata():
    meta = {"host": "https://justeddy1-kataa-demo.static.hf.space", "subdomain": "justeddy1-kataa-demo"}
    assert csp.space_host(meta) == "justeddy1-kataa-demo.static.hf.space"


def test_fetch_sends_no_credentials(monkeypatch):
    seen = {}

    class Resp:
        status = 200
        def read(self):
            return b""
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        return Resp()

    monkeypatch.setattr(csp.urllib.request, "urlopen", fake_urlopen)
    assert csp.fetch("https://huggingface.co/x")[0] == 200
    assert "authorization" not in seen["headers"] and "cookie" not in seen["headers"]
