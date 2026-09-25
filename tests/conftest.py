"""Fixtures compartidas: un dataset sintético pequeño generado una sola vez por sesión."""

import json

import polars as pl
import pytest

import generate_data

SMALL_DATASET_ARGS = ["--sales-rows", "4000", "--customers", "500", "--products", "80", "--seed", "7"]


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw")
    assert generate_data.main(["--output-dir", str(out), *SMALL_DATASET_ARGS]) == 0
    return out


@pytest.fixture(scope="session")
def manifest(small_dataset):
    return json.loads((small_dataset / "_manifest.json").read_text(encoding="utf-8"))


def read_sheet(directory, name: str) -> pl.DataFrame:
    return pl.read_csv(directory / f"{name}.csv", infer_schema=False)


def sheet(header: list[str], rows: list[list]) -> pl.DataFrame:
    """Construye una "hoja" de texto como la exportaría Google Sheets."""
    return pl.DataFrame([dict(zip(header, row)) for row in rows], schema={h: pl.String for h in header})
