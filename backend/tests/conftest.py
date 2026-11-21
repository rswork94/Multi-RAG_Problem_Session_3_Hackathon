"""Test configuration: always offline (no API spend), hashing embeddings (no 1.3 GB model), throwaway data dir.

Env vars must be set before `app` is imported, because app.config reads them at import time.
"""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="multirag-test-")
os.environ["DATA_DIR"] = _TMP
os.environ["MULTIRAG_FAKE_EMBED"] = "1"
os.environ["MULTIRAG_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("ANTHROPIC_API_KEY", None)

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def mini_corpus(tmp_path_factory):
    """Render two documents from the real tax spec (a notification + the rules extract) and ingest them."""
    from corpus_lib import RENDERERS, assembled_documents, doc_filename

    from app.engines.ingest_graph import run_ingest

    out = tmp_path_factory.mktemp("corpus")
    docs = assembled_documents("tax")
    files = []
    for d in docs:
        p = out / doc_filename(d)
        RENDERERS[d["format"]](d, p)
        files.append({"path": str(p), "filename": p.name, "area": "tax"})
    res = run_ingest(files, use_llm=False, confirm_cost=False)
    return {"docs": docs, "result": res}
