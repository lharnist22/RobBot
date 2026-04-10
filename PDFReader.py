import re
import json
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_pdf_text(text: str) -> str:
    # keep valid utf-8 only
    text = text.encode("utf-8", "ignore").decode()

    # normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # remove random single-letter lines (common PDF artifacts)
    text = re.sub(r"\n[A-Za-z]\n", "\n", text)

    # remove simple page footer/header patterns
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.MULTILINE)

    # collapse spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:  # skip None / empty pages
            parts.append(t)
    return "\n".join(parts)

def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "pdf"

def main():
    ingest_dir = Path("toIngest")
    out_path = Path("knowledge_base.jsonl")

    if not ingest_dir.exists():
        raise FileNotFoundError(f"Missing folder: {ingest_dir}")

    pdf_paths = sorted(ingest_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in {ingest_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )

    records = []
    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path}")
        full_text = extract_pdf_text(str(pdf_path))
        full_text = clean_pdf_text(full_text)

        if not full_text:
            print(f"Skipping empty extract: {pdf_path}")
            continue

        chunks = splitter.split_text(full_text)
        source_slug = slugify(pdf_path.stem)
        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"{source_slug}_chunk_{i}",
                "text": chunk,
                "source": str(pdf_path).replace("\\", "/"),
            })
        print(f"  Added {len(chunks)} chunks")

    if not records:
        raise RuntimeError("No chunks were produced from the PDFs.")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} chunks to {out_path}")

if __name__ == "__main__":
    main()
