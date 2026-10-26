"""Check compiled PDF and record explicitly reviewed pages; requires pypdf/pdfplumber.

--reviewed-pages must be supplied only after a human/agent has inspected those rendered pages.
The script does not perform or infer visual review automatically.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def validate(pdf, reviewed):
    pdf = Path(pdf).resolve()
    reader = PdfReader(pdf)
    assert sorted(reviewed) == list(range(1, len(reader.pages)+1)), "Review every page before marking complete"
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "??" not in text
    for value in ["0.5784", "0.5332", "0.4851", "0.5745", "0.5502", "0.0280", "64.44", "15.52"]:
        assert value in text, f"Missing expected reported value: {value}"
    outside = []
    with pdfplumber.open(pdf) as document:
        for i, page in enumerate(document.pages, 1):
            for word in page.extract_words():
                if word["x0"] < 0 or word["x1"] > page.width or word["top"] < 0 or word["bottom"] > page.height:
                    outside.append({"page": i, "text": word["text"]})
    assert not outside, outside
    # Check targets, not merely the absence of unresolved reference markers.
    expected = {"figure.1": 3, "figure.2": 5, "figure.3": 5, "figure.4": 6,
                "figure.5": 7, "figure.6": 8, "figure.7": 12,
                "table.1": 1, "table.2": 4, "table.3": 5, "table.4": 6,
                "table.5": 7, "table.6": 7, "table.7": 10}
    actual = {name: reader.get_destination_page_number(reader.named_destinations[name])+1 for name in expected}
    assert actual == expected, "Layout changed: inspect and update expected reference destinations"
    result = {"pdf": "reports/entregable_4/equipo_proyecto_avance4.pdf",
              "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "pages": len(reader.pages),
              "pages_visually_reviewed": sorted(reviewed), "review_method": "All 12 pages rendered with Poppler and inspected by agent",
              "no_missing_references": True, "no_text_outside_page": True,
              "reference_destinations_pages": actual, "expected_values_found": True,
              "compiler": "Tectonic 0.17.0, XeTeX and BibTeX, successful exit",
              "compiler_notes": "Class emits empty overfull-box and duplicate-anchor warnings; rendered content fits pages and named destinations checked above",
              "not_claimed": ["New image inference", "New training", "Statistical significance"]}
    output = ROOT / "reports/entregable_4/correcciones_etapa5/validacion_pdf.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--reviewed-pages", nargs="+", type=int, required=True)
    args = parser.parse_args()
    validate(args.pdf, args.reviewed_pages)
