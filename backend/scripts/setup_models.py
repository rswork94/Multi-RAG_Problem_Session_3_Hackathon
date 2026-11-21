"""Pre-download every model and data file the app needs, so demos can run with HF_HUB_OFFLINE=1.

Usage:  .venv/bin/python scripts/setup_models.py
Downloads: BAAI/bge-large-en-v1.5 (~1.3 GB), BAAI/bge-reranker-base (~1.1 GB), NLTK data for unstructured.
"""
from __future__ import annotations

import os
import time

# python.org builds ship without CA certs; NLTK downloads fail with CERTIFICATE_VERIFY_FAILED otherwise.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass


def main() -> None:
    t0 = time.time()
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print("Downloading BAAI/bge-large-en-v1.5 ...", flush=True)
    m = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    print(f"  dim={m.get_embedding_dimension()} max_seq={m.max_seq_length}", flush=True)

    print("Downloading BAAI/bge-reranker-base ...", flush=True)
    CrossEncoder("BAAI/bge-reranker-base", device="cpu")

    print("Downloading NLTK data used by unstructured ...", flush=True)
    try:
        from unstructured.nlp.tokenize import download_nltk_packages
        download_nltk_packages()
    except Exception as e:  # older/newer unstructured layouts
        import nltk
        for pkg in ("punkt_tab", "averaged_perceptron_tagger_eng"):
            nltk.download(pkg, quiet=True)
        print(f"  (fallback nltk download used: {e})")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
