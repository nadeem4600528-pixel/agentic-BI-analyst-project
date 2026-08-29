"""PDF-ready HTML export without adding an unconfirmed dependency."""
import html
import json
from typing import Any, Mapping


def export_html(report: Mapping[str, Any]) -> str:
    body = html.escape(json.dumps(report, indent=2, default=str))
    return f"<!doctype html><html><head><meta charset='utf-8'><title>BI Report</title></head><body><h1>Agentic BI Report</h1><pre>{body}</pre></body></html>"


def export_pdf(report: Mapping[str, Any]) -> bytes:
    """Create a small valid PDF without requiring an external PDF package."""
    lines = json.dumps(report, indent=2, default=str).splitlines()[:55]
    escaped = [line[:110].replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    content = "BT /F1 8 Tf 40 760 Td " + " 0 -11 Td ".join(f"({line}) Tj" for line in escaped) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        f"<< /Length {len(content.encode('latin-1', errors='replace'))} >>\nstream\n{content}\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf)); pdf.extend(f"{number} 0 obj\n{obj}\nendobj\n".encode("latin-1", errors="replace"))
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]: pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return bytes(pdf)

