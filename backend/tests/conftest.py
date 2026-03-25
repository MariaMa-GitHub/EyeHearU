"""Pytest fixtures and shared setup."""

import pytest
from app.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Avoid env bleed between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
