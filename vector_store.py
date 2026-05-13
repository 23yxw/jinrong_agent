"""Qdrant setup and ingestion helpers."""

import hashlib
import json
import re
from pathlib import Path

from config import QDRANT_COLLECTION, QDRANT_URL
from model_factory import get_embedding_model
from pdf_pipeline import extract_metadata_from_filename
from runtime_checks import assert_runtime_ready

TEXT_CHUNK_SIZE = 600
TEXT_CHUNK_OVERLAP = 100

NOISY_PAGE_ROLES = {
    "table_of_contents",
    "financial_index",
    "audit_report",
    "internal_control",
    "certification",
    "website_disclosure",
    "exhibit_index",
}

STRUCTURED_PAGE_ROLES = {
    "risk_factors",
    "cash_flow_statement",
    "income_statement",
    "balance_sheet",
    "shareholders_equity",
    "gross_margin_discussion",
    "md_and_a",
}


def build_vector_store():
    assert_runtime_ready(
        stage="vector_store.build_vector_store",
        packages=["langchain_qdrant", "qdrant_client"],
    )
    from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
    from qdrant_client import QdrantClient

    embeddings = get_embedding_model()
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    client = QdrantClient(url=QDRANT_URL)

    collection_exists = False
    try:
        collection_exists = bool(client.collection_exists(QDRANT_COLLECTION))
    except Exception:  # noqa: BLE001
        # Fallback for older client versions that may not expose collection_exists.
        try:
            client.get_collection(QDRANT_COLLECTION)
            collection_exists = True
        except Exception:  # noqa: BLE001
            collection_exists = False

    if collection_exists:
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=QDRANT_COLLECTION,
            url=QDRANT_URL,
            retrieval_mode=RetrievalMode.HYBRID,
        )

    # Auto-create collection when it does not exist yet.
    return QdrantVectorStore.from_documents(
        documents=[],
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        collection_name=QDRANT_COLLECTION,
        url=QDRANT_URL,
        retrieval_mode=RetrievalMode.HYBRID,
        force_recreate=False,
    )


def compute_file_hash(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def extract_page_number(file_path: Path) -> int:
    match = re.search(r"page_(\d+)", str(file_path))
    return int(match.group(1)) if match else 0


def get_processed_hashes(vector_store) -> set[str]:
    processed_hashes: set[str] = set()
    offset = None

    while True:
        points, offset = vector_store.client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10_000,
            with_payload=True,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            payload = point.payload or {}
            file_hash = payload.get("file_hash")
            if file_hash is None and isinstance(payload.get("metadata"), dict):
                file_hash = payload["metadata"].get("file_hash")
            if file_hash:
                processed_hashes.add(file_hash)

        if offset is None:
            break

    return processed_hashes


def _load_file_content(file_path: Path) -> str | None:
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


def _load_tables(file_path: Path) -> list[dict] | None:
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, list):
        return None
    return [item for item in data if isinstance(item, dict)]


def _split_text_chunks(text: str, chunk_size: int = TEXT_CHUNK_SIZE, chunk_overlap: int = TEXT_CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            split_at = max(
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("。", start, end),
                text.rfind(" ", start, end),
            )
            if split_at > start + max(chunk_size // 3, 200):
                end = split_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def _normalize_match_text(text: str) -> str:
    return " ".join((text or "").replace("\xa0", " ").lower().split())


def _extract_primary_heading(text: str) -> str | None:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None


def _classify_page_role(text: str) -> str:
    normalized = _normalize_match_text(text)
    if not normalized:
        return "general"
    if "table of contents" in normalized:
        return "table_of_contents"
    if "index to consolidated financial statements" in normalized:
        return "financial_index"
    if "report of independent registered public accounting firm" in normalized:
        return "audit_report"
    if "internal control over financial reporting" in normalized:
        return "internal_control"
    if (
        "certifications of chief executive officer" in normalized
        or "i have reviewed this annual report" in normalized
        or "certify that:" in normalized
    ):
        return "certification"
    if "investor relations website" in normalized or "sec-filings/default.aspx" in normalized:
        return "website_disclosure"
    if "exhibit index" in normalized or "form s-8" in normalized:
        return "exhibit_index"
    if "item 1a. risk factors" in normalized or "risk factors" in normalized:
        return "risk_factors"
    if "consolidated statements of cash flows" in normalized:
        return "cash_flow_statement"
    if (
        "consolidated statements of operations" in normalized
        or "consolidated statements of comprehensive income" in normalized
    ):
        return "income_statement"
    if "consolidated balance sheets" in normalized:
        return "balance_sheet"
    if "consolidated statements of shareholders' equity" in normalized:
        return "shareholders_equity"
    if "gross margin" in normalized:
        return "gross_margin_discussion"
    if (
        "management's discussion and analysis" in normalized
        or "results of operations" in normalized
        or "financial condition" in normalized
    ):
        return "md_and_a"
    return "general"


def _split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = stripped[3:].strip() or None
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return [(heading, body) for heading, body in sections if body]


def _split_page_chunks(text: str, page_role: str) -> list[tuple[str, str | None]]:
    stripped = (text or "").strip()
    if not stripped:
        return []

    primary_heading = _extract_primary_heading(stripped)

    if page_role in NOISY_PAGE_ROLES:
        return [(stripped, primary_heading)]

    if page_role in STRUCTURED_PAGE_ROLES:
        chunk_size = 1000 if page_role in {"risk_factors", "md_and_a", "gross_margin_discussion"} else 1400
        chunk_overlap = 120 if page_role in {"risk_factors", "md_and_a"} else 80
        sections = _split_markdown_sections(stripped) or [(primary_heading, stripped)]
        structured_chunks: list[tuple[str, str | None]] = []
        for section_heading, section_text in sections:
            for chunk in _split_text_chunks(section_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                structured_chunks.append((chunk, section_heading or primary_heading))
        if structured_chunks:
            return structured_chunks

    return [(chunk, primary_heading) for chunk in _split_text_chunks(stripped)]


def ingest_file(file_path: Path, processed_hashes: set[str], vector_store):
    assert_runtime_ready(
        stage="vector_store.ingest_file",
        packages=["langchain_core"],
    )
    from langchain_core.documents import Document

    file_hash = compute_file_hash(file_path)
    if file_hash in processed_hashes:
        return False

    path_str = str(file_path)
    if "markdown" in path_str:
        content_type = "text"
        doc_name = file_path.name
    elif "tables" in path_str:
        content_type = "table"
        doc_name = file_path.name
    elif "images_desc" in path_str or "images" in path_str:
        content_type = "image_description"
        doc_name = file_path.name
    else:
        content_type = "unknown"
        doc_name = file_path.name

    path_parts = set(file_path.parts)
    company_hint = None
    if "markdown" in path_parts or "tables" in path_parts or "images_desc" in path_parts or "images" in path_parts:
        if len(file_path.parents) >= 2:
            company_hint = file_path.parents[1].name

    metadata = extract_metadata_from_filename(company_hint or doc_name)
    if company_hint:
        file_name_meta = extract_metadata_from_filename(doc_name)
        for key, value in file_name_meta.items():
            if metadata.get(key) is None and value is not None:
                metadata[key] = value

    metadata.update(
        {
            "content_type": content_type,
            "file_hash": file_hash,
            "source_file": doc_name,
        }
    )

    documents: list[Document] = []
    if content_type == "text":
        content = _load_file_content(file_path)
        if not content:
            return False
        pages = content.split("<!-- page break -->")
        for idx, page in enumerate(pages, start=1):
            page = page.strip()
            if not page:
                continue
            page_role = _classify_page_role(page)
            page_chunks = _split_page_chunks(page, page_role)
            for chunk_idx, (chunk, section_heading) in enumerate(page_chunks, start=1):
                page_metadata = metadata.copy()
                page_metadata["page"] = idx
                page_metadata["page_role"] = page_role
                page_metadata["is_noise_prone"] = page_role in NOISY_PAGE_ROLES
                page_metadata["section_heading"] = section_heading
                page_metadata["chunk_index"] = chunk_idx
                page_metadata["chunk_count"] = len(page_chunks)
                documents.append(Document(page_content=chunk, metadata=page_metadata))
    elif content_type == "table":
        tables = _load_tables(file_path)
        if not tables:
            return False
        for table_idx, table in enumerate(tables, start=1):
            table_content = str(table.get("content", "")).strip()
            if not table_content:
                continue
            table_role = _classify_page_role(table_content)
            table_metadata = metadata.copy()
            table_metadata["page"] = int(table.get("page", 0) or 0)
            table_metadata["page_role"] = table_role
            table_metadata["is_noise_prone"] = table_role in NOISY_PAGE_ROLES
            table_metadata["section_heading"] = _extract_primary_heading(table_content)
            table_metadata["table_id"] = table.get("table_id", f"table_{table_idx}")
            table_metadata["chunk_index"] = table_idx
            table_metadata["chunk_count"] = len(tables)
            documents.append(Document(page_content=table_content, metadata=table_metadata))
    else:
        content = _load_file_content(file_path)
        if not content:
            return False
        file_metadata = metadata.copy()
        file_metadata["page"] = extract_page_number(file_path)
        file_metadata["page_role"] = _classify_page_role(content)
        file_metadata["is_noise_prone"] = file_metadata["page_role"] in NOISY_PAGE_ROLES
        file_metadata["section_heading"] = _extract_primary_heading(content)
        documents.append(Document(page_content=content, metadata=file_metadata))

    if not documents:
        return False

    vector_store.add_documents(documents)
    processed_hashes.add(file_hash)
    return True


def ingest_all(data_dir: str):
    root = Path(data_dir)
    vector_store = build_vector_store()
    processed = get_processed_hashes(vector_store)

    allowed_dir_names = {"markdown", "tables", "images_desc", "images"}
    all_files: list[Path] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".md", ".json"}:
            continue
        if not any(part in allowed_dir_names for part in file_path.parts):
            continue
        all_files.append(file_path)

    success = 0
    for file_path in all_files:
        if ingest_file(file_path, processed, vector_store):
            success += 1

    print(f"Ingestion done. Added: {success}, scanned: {len(all_files)}")
