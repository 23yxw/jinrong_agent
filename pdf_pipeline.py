"""PDF processing pipeline using Docling."""

import base64
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from config import OUTPUT_DIR
from model_factory import get_chat_model
from runtime_checks import assert_runtime_ready

load_dotenv()

IMAGE_DESCRIPTION_PROMPT = """
You are a financial analyst expert. Analyze this image from a financial document and provide a detailed description.

Your description must include:
1. The visualization type (bar chart, line chart, pie chart, table, etc.)
2. The visible title or heading
3. Specific data points and numbers that are clearly visible
4. Trends, patterns, and key comparisons
5. Labels, legends, and annotations
6. Time period (if available)
7. Units of measurement

Be precise with numbers and avoid guessing when values are not visible.
Format the response in markdown.
""".strip()


def extract_metadata_from_filename(filename: str) -> dict:
    stem = Path(filename).stem.lower()
    stem = stem.replace("_tables", "").replace("_description", "")
    normalized = stem.replace("-", "_")
    parts = [token for token in re.split(r"[\s_]+", normalized) if token]

    company_name = parts[0] if parts else None
    doc_type = None
    fiscal_quarter = None
    fiscal_year = None

    for token in parts:
        if token in {"10k", "10-k"}:
            doc_type = "10-k"
        elif token in {"10q", "10-q"}:
            doc_type = "10-q"
        elif token in {"q1", "q2", "q3", "q4"}:
            fiscal_quarter = token
        elif token.isdigit() and len(token) == 4:
            fiscal_year = int(token)

    return {
        "company_name": company_name,
        "doc_type": doc_type,
        "fiscal_quarter": fiscal_quarter,
        "fiscal_year": fiscal_year,
    }


def create_output_dirs(company_name: str, base_dir: Path | str = OUTPUT_DIR) -> dict:
    base = Path(base_dir)
    dirs = {
        "markdown": base / company_name / "markdown",
        "images": base / company_name / "images",
        "images_desc": base / company_name / "images_desc",
        "tables": base / company_name / "tables",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def build_converter():
    assert_runtime_ready(
        stage="pdf_pipeline.build_converter",
        packages=["docling", "docling_core"],
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.images_scale = 2
    options.generate_picture_images = True
    options.generate_page_images = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def convert_pdf(pdf_path: str):
    converter = build_converter()
    return converter.convert(pdf_path)


def save_markdown(markdown_content: str, markdown_dir: Path | str, source_name: str) -> Path:
    output_dir = Path(markdown_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{Path(source_name).stem}.md"
    output_file.write_text(markdown_content, encoding="utf-8")
    return output_file


def save_tables(tables: list[dict], tables_dir: Path | str, source_name: str) -> Path:
    output_dir = Path(tables_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{Path(source_name).stem}_tables.json"
    output_file.write_text(
        json.dumps(tables, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file


def save_page_images(doc_result, images_dir: str):
    assert_runtime_ready(
        stage="pdf_pipeline.save_page_images",
        packages=["docling_core"],
    )
    from docling_core.types.doc import PictureItem

    images_path = Path(images_dir)
    images_path.mkdir(parents=True, exist_ok=True)
    pages_to_save = set()

    for item in doc_result.document.iterate_items():
        element = item[0]
        if isinstance(element, PictureItem):
            image = element.get_image(doc_result.document)
            if image and image.size[0] > 500 and image.size[1] > 500:
                page_no = element.prov[0].page_no if element.prov else None
                if page_no is not None:
                    pages_to_save.add(page_no)

    total_pages = len(doc_result.document.pages)
    for page_no in pages_to_save:
        page_idx = page_no - 1 if page_no > 0 else page_no
        if page_idx < 0 or page_idx >= total_pages:
            continue
        page = doc_result.document.pages[page_idx]
        if page.image and page.image.pil_image:
            output_file = images_path / f"page_{page_no}.png"
            page.image.pil_image.save(output_file, "PNG")


def extract_tables_with_context(markdown_content: str):
    lines = markdown_content.split("\n")
    tables = []
    current_page = 1
    table_num = 1
    i = 0

    while i < len(lines):
        if "<!-- page break -->" in lines[i]:
            current_page += 1
            i += 1
            continue

        if lines[i].startswith("|") and lines[i].count("|") > 1:
            table_lines = []
            table_start = i
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1

            context_start = max(0, table_start - 2)
            context_lines = lines[context_start:table_start]
            content = "\n".join(context_lines) + "\n\n" + "\n".join(table_lines)
            tables.append(
                {
                    "content": content.strip(),
                    "table_id": f"table_{table_num}",
                    "page": current_page,
                }
            )
            table_num += 1
        else:
            i += 1

    return tables


def process_all_pdfs(data_dir: str = "data/rag_data"):
    pdf_paths = list(Path(data_dir).rglob("*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found in {data_dir}")
        return

    converter = build_converter()
    for pdf_file in pdf_paths:
        try:
            metadata = extract_metadata_from_filename(pdf_file.name)
            company_name = metadata.get("company_name") or "unknown"
            dirs = create_output_dirs(company_name)

            result = converter.convert(str(pdf_file))
            markdown = result.document.export_to_markdown(
                page_break_placeholder="<!-- page break -->"
            )
            save_markdown(markdown, dirs["markdown"], pdf_file.name)
            save_page_images(result, str(dirs["images"]))

            tables = extract_tables_with_context(markdown)
            save_tables(tables, dirs["tables"], pdf_file.name)
            print(f"Done: {pdf_file.name} ({len(tables)} tables)")
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {pdf_file.name}: {exc}")


def _image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _build_vision_model():
    assert_runtime_ready(
        stage="pdf_pipeline._build_vision_model",
        packages=["langchain_core"],
    )
    return get_chat_model()


def generate_image_description(image_path: Path) -> str:
    assert_runtime_ready(
        stage="pdf_pipeline.generate_image_description",
        packages=["langchain_core"],
    )
    from langchain_core.messages import HumanMessage

    model = _build_vision_model()
    image_base64 = _image_to_base64(image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{image_base64}",
            },
        ]
    )
    response = model.invoke([message])
    return str(response.content)


def process_all_image_descriptions(output_root: str = "output", overwrite: bool = False):
    root = Path(output_root)
    if not root.exists():
        print(f"Output root not found: {root}")
        return

    total = 0
    done = 0
    skipped = 0
    failed = 0

    for company_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images_dir = company_dir / "images"
        desc_dir = company_dir / "images_desc"
        if not images_dir.exists():
            continue

        desc_dir.mkdir(parents=True, exist_ok=True)
        image_files = sorted(images_dir.glob("*.png"))
        for image_file in image_files:
            total += 1
            output_file = desc_dir / f"{image_file.stem}_description.md"
            if output_file.exists() and not overwrite:
                skipped += 1
                continue
            try:
                description = generate_image_description(image_file)
                output_file.write_text(description, encoding="utf-8")
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"Failed: {image_file} -> {exc}")

    print(
        f"Image description summary: total={total}, generated={done}, skipped={skipped}, failed={failed}"
    )
