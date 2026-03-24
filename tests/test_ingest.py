"""Test per il modulo app.ingest."""

import pytest
from pathlib import Path

from haystack import Document
from app.ingest import ingest, UnsupportedFormatError

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TestIngestCSV:
    def test_basic(self):
        docs = ingest(FIXTURES / "test.csv", content_column="testo")
        # La riga vuota (indice 2) deve essere saltata
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)

    def test_content(self):
        docs = ingest(FIXTURES / "test.csv", content_column="testo")
        assert docs[0].content == "Primo documento di test"

    def test_mandatory_meta(self):
        docs = ingest(FIXTURES / "test.csv", content_column="testo")
        for doc in docs:
            assert "id" in doc.meta
            assert "source" in doc.meta
            assert "row_index" in doc.meta

    def test_metadata_columns(self):
        docs = ingest(
            FIXTURES / "test.csv",
            content_column="testo",
            metadata_columns=["autore"],
        )
        assert "autore" in docs[0].meta
        assert "data" not in docs[0].meta

    def test_all_metadata_columns_by_default(self):
        docs = ingest(FIXTURES / "test.csv", content_column="testo")
        assert "autore" in docs[0].meta
        assert "data" in docs[0].meta

    def test_extra_metadata(self):
        docs = ingest(
            FIXTURES / "test.csv",
            content_column="testo",
            extra_metadata={"language": "it", "category": "test"},
        )
        for doc in docs:
            assert doc.meta["language"] == "it"
            assert doc.meta["category"] == "test"

    def test_missing_content_column(self):
        with pytest.raises(ValueError, match="Colonne disponibili"):
            ingest(FIXTURES / "test.csv", content_column="colonna_inesistente")

    def test_empty_rows_skipped(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            docs = ingest(FIXTURES / "test.csv", content_column="testo")
        assert len(docs) == 3
        assert any("saltata" in r.message for r in caplog.records)

    def test_id_format(self):
        docs = ingest(FIXTURES / "test.csv", content_column="testo")
        assert docs[0].meta["id"].startswith("test_row_")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

class TestIngestJSON:
    def test_basic(self):
        docs = ingest(FIXTURES / "test.json", content_column="testo")
        # Riga vuota saltata
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    def test_content(self):
        docs = ingest(FIXTURES / "test.json", content_column="testo")
        assert docs[0].content == "Primo articolo JSON"

    def test_mandatory_meta(self):
        docs = ingest(FIXTURES / "test.json", content_column="testo")
        for doc in docs:
            assert "id" in doc.meta
            assert "source" in doc.meta

    def test_missing_content_column(self):
        with pytest.raises(ValueError, match="Colonne disponibili"):
            ingest(FIXTURES / "test.json", content_column="inesistente")


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

class TestIngestJSONL:
    def test_basic(self):
        docs = ingest(FIXTURES / "test.jsonl", content_column="testo")
        assert len(docs) == 2

    def test_content(self):
        docs = ingest(FIXTURES / "test.jsonl", content_column="testo")
        assert docs[0].content == "Prima riga JSONL"
        assert docs[1].content == "Seconda riga JSONL"

    def test_metadata_columns(self):
        docs = ingest(
            FIXTURES / "test.jsonl",
            content_column="testo",
            metadata_columns=["categoria"],
        )
        assert docs[0].meta["categoria"] == "A"


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------

class TestIngestTXT:
    def test_basic(self):
        docs = ingest(FIXTURES / "test.txt")
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_content_not_empty(self):
        docs = ingest(FIXTURES / "test.txt")
        assert docs[0].content.strip()

    def test_mandatory_meta(self):
        docs = ingest(FIXTURES / "test.txt")
        assert "id" in docs[0].meta
        assert "source" in docs[0].meta

    def test_id_is_stem(self):
        docs = ingest(FIXTURES / "test.txt")
        assert docs[0].meta["id"] == "test"

    def test_extra_metadata(self):
        docs = ingest(FIXTURES / "test.txt", extra_metadata={"language": "it"})
        assert docs[0].meta["language"] == "it"

    def test_latin1_fallback(self, tmp_path):
        latin1_file = tmp_path / "latin1.txt"
        latin1_file.write_bytes("Testo con accenti: àèìòù".encode("latin-1"))
        docs = ingest(latin1_file)
        assert len(docs) == 1
        assert "accenti" in docs[0].content


# ---------------------------------------------------------------------------
# Errori comuni
# ---------------------------------------------------------------------------

class TestIngestErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ingest("file_inesistente.csv", content_column="testo")

    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "doc.xyz"
        f.write_text("contenuto")
        with pytest.raises(UnsupportedFormatError):
            ingest(f)

    def test_unsupported_format_message(self, tmp_path):
        f = tmp_path / "doc.xyz"
        f.write_text("contenuto")
        with pytest.raises(UnsupportedFormatError, match=".csv"):
            ingest(f)
