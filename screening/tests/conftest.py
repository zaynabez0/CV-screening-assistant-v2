"""Shared pytest configuration for the screening suite."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_embedding_model: requires BAAI/bge-m3 to be downloadable or already cached",
    )


@pytest.fixture(scope="session")
def embedding_model():
    """The real BGE-M3 model, or a skip with the reason why it isn't available.

    Downloads ~2 GB from HuggingFace on first use and caches it. Session-scoped so the
    load cost is paid once for the whole run.
    """
    try:
        from screening.stage2_ranking.embeddings import get_embedding_model
        return get_embedding_model()
    except Exception as exc:  # network blocked, offline, out of disk
        pytest.skip(f"BAAI/bge-m3 unavailable ({type(exc).__name__}: {str(exc)[:120]})")
