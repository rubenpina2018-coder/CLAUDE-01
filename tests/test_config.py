"""Tests de la lectura de configuración (.env)."""

import os

import pytest

from analytics_engine.config import load_dotenv

KEYS = ("DW_HOST", "DW_PORT", "DW_PASSWORD")


@pytest.fixture
def clean_env(monkeypatch):
    # setenv + delenv registra el estado original: al terminar el test se restaura (no hay fugas)
    for key in KEYS:
        monkeypatch.setenv(key, "placeholder")
        monkeypatch.delenv(key)
    return monkeypatch


def test_load_dotenv_handles_windows_bom_quotes_and_comments(tmp_path, clean_env):
    env = tmp_path / ".env"
    env.write_bytes('DW_HOST=db.example.com\n# comentario\n\nDW_PORT = 6543\nDW_PASSWORD="p@ss=word"\n'
                    .encode("utf-8-sig"))  # con BOM, como guardan algunos editores de Windows
    load_dotenv(env)
    assert os.environ["DW_HOST"] == "db.example.com"
    assert os.environ["DW_PORT"] == "6543"
    assert os.environ["DW_PASSWORD"] == "p@ss=word"


def test_existing_environment_variables_take_precedence(tmp_path, clean_env):
    env = tmp_path / ".env"
    env.write_text("DW_HOST=desde_fichero\n", encoding="utf-8")
    clean_env.setenv("DW_HOST", "desde_entorno")
    load_dotenv(env)
    assert os.environ["DW_HOST"] == "desde_entorno"
