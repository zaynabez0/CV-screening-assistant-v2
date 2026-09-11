"""Test-wide environment isolation.

`screening/common/llm.py` reads its configuration from the environment and loads the
project's `.env` at import. Without this fixture the developer's own `.env` — or a
leftover variable in their shell — would decide what the suite sees, and the model id
recorded against a run would differ from machine to machine. Pinned here instead, so a
fresh clone with no `.env` at all gives the same answers as a configured one.

Tests that care about a specific value set it themselves; a module-level fixture runs
after this one and can still unset these to test the "not configured" paths.
"""

import pytest

from screening.common import llm

TEST_MODEL = "test-model"
TEST_API_KEY = "sk-test"


@pytest.fixture(autouse=True)
def llm_environment(monkeypatch):
    monkeypatch.setenv(llm.API_KEY_ENV_VAR, TEST_API_KEY)
    monkeypatch.setenv(llm.MODEL_ENV_VAR, TEST_MODEL)
    monkeypatch.delenv(llm.BASE_URL_ENV_VAR, raising=False)
    llm.reset_client()
    yield
    llm.reset_client()
