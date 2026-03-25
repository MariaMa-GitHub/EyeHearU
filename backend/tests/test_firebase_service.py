"""Coverage for firebase_service with mocks."""

from unittest.mock import MagicMock, patch

import pytest

from app.services import firebase_service as fb


@pytest.fixture(autouse=True)
def reset_firebase_state():
    fb._db = None
    yield
    fb._db = None


def test_init_firebase_first_time():
    mock_db = MagicMock()
    mock_cred = MagicMock()
    with (
        patch("firebase_admin._apps", {}),
        patch("firebase_admin.credentials.Certificate", return_value=mock_cred),
        patch("firebase_admin.initialize_app"),
        patch("firebase_admin.firestore.client", return_value=mock_db),
        patch("app.services.firebase_service.get_settings") as gs,
    ):
        gs.return_value.firebase_credentials_path = "c.json"
        gs.return_value.firebase_project_id = "proj"
        db = fb.init_firebase()
    assert db is mock_db
    assert fb._db is mock_db


def test_init_firebase_already_initialized():
    mock_db = MagicMock()
    mock_cred = MagicMock()
    fake_apps = {"x": 1}
    with (
        patch("firebase_admin._apps", fake_apps),
        patch("firebase_admin.credentials.Certificate", return_value=mock_cred),
        patch("firebase_admin.initialize_app") as init,
        patch("firebase_admin.firestore.client", return_value=mock_db),
        patch("app.services.firebase_service.get_settings") as gs,
    ):
        gs.return_value.firebase_credentials_path = "c.json"
        gs.return_value.firebase_project_id = "proj"
        db = fb.init_firebase()
    init.assert_not_called()
    assert db is mock_db


def test_get_db_initializes_when_none():
    fb._db = None
    mock_db = MagicMock()
    with patch.object(fb, "init_firebase", return_value=mock_db) as ini:
        db = fb.get_db()
    ini.assert_called_once()
    assert db is mock_db


def test_get_db_returns_cached():
    cached = MagicMock()
    fb._db = cached
    with patch.object(fb, "init_firebase") as ini:
        db = fb.get_db()
    ini.assert_not_called()
    assert db is cached
    fb._db = None


def test_save_translation():
    mock_db = MagicMock()
    col = MagicMock()
    mock_db.collection.return_value = col
    with patch.object(fb, "get_db", return_value=mock_db):
        fb.save_translation("sess1", {"predicted_sign": "hello", "confidence": 0.9})
    col.add.assert_called_once()
    args = col.add.call_args[0][0]
    assert args["session_id"] == "sess1"
    assert args["predicted_sign"] == "hello"


def test_get_translation_history():
    doc = MagicMock()
    doc.to_dict.return_value = {"a": 1}
    stream = iter([doc])
    mock_q = MagicMock()
    mock_q.where.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.limit.return_value = mock_q
    mock_q.stream.return_value = stream
    mock_db = MagicMock()
    mock_db.collection.return_value = mock_q
    with patch.object(fb, "get_db", return_value=mock_db):
        rows = fb.get_translation_history("sess1", limit=10)
    assert rows == [{"a": 1}]
