import sys, uuid
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import TableItem, TextItem

NS = uuid.UUID("f3c1e5d0-8b6e-4f9a-9c2b-1e5d0f8b6e4f")

def main():
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    pdf_files = list(src.glob("*.pdf"))
    for pdf in sorted(pdf_files):
        print(f"parsing {pdf.name}...")
        doc = converter.convert(pdf).document
        out = []
        for i, (item, _) in enumerate(doc.iterate_items()):
            if isinstance(item, TableItem):
                cap = item.caption_text(doc) or ""
                text = (cap + "\n" + item.export_to_markdown(doc)).strip()
            elif isinstance(item, TextItem):
                text = (item.text or "").strip()
            else:
                continue
            if not text:
                continue
            cid = uuid.uuid5(NS, f"{pdf.stem}|{i}|{text}")
            out.append(f"ID: {cid}\n{text}\n")
        out_text = "\n".join(out).replace("\x00", "")
        (dest / f"{pdf.stem}.md").write_text(out_text)
    print("done ->", dest)

if __name__ == "__main__":
    main()
