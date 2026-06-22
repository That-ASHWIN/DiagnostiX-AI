"""Build a downloadable PDF of a DiagnostiX AI diagnosis.

Uses fpdf2 (pure Python, no system dependencies) so it works out of the box on
Streamlit Community Cloud. fpdf2's built-in fonts only support latin-1, so all
text is sanitised through ``_safe`` before being written.
"""
from datetime import datetime

from fpdf import FPDF


def _safe(text):
    """Drop characters the built-in PDF fonts cannot encode (latin-1 only)."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _match_strength_label(score):
    if score >= 0.75:
        return "Strong match"
    if score >= 0.40:
        return "Moderate match"
    return "Weak match - review the alternatives"


def build_report_pdf(inputs, result):
    """Return the diagnosis report as PDF bytes.

    ``inputs`` is an ordered dict of the values the user entered and ``result``
    is the dict returned by ``diagnosis.predict_diagnosis``.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Header -----------------------------------------------------------
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, _safe("DiagnostiX AI"))
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, _safe("Electronic Device Fault Diagnosis Report"))
    pdf.ln(7)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 6, _safe("Generated on " + datetime.now().strftime("%d %b %Y, %H:%M")))
    pdf.ln(11)
    pdf.set_text_color(0, 0, 0)

    def heading(title):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, _safe(title))
        pdf.ln(9)
        pdf.set_font("Helvetica", "", 11)

    def field(label, value):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(62, 7, _safe(label))
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, _safe(str(value)))

    # --- Inputs -----------------------------------------------------------
    heading("Device & usage details")
    for label, value in inputs.items():
        field(label + ":", value)

    # --- Diagnosis --------------------------------------------------------
    heading("Diagnosis")
    confidence = result.get("confidence", 0) or 0
    field("Most likely component:", result.get("fault", "-"))
    field(
        "Symptom match:",
        f"{_match_strength_label(confidence)} ({confidence * 100:.0f}%)",
    )
    field("Severity:", result.get("severity", "-"))
    field(
        "Estimated repair cost:",
        f"Rs {result.get('estimated_cost_inr', 0):,.0f}",
    )
    field(
        "Estimated repair time:",
        f"{result.get('estimated_time_hours', 0):.1f} hours",
    )

    # --- Recommended solution --------------------------------------------
    steps = result.get("solution_steps") or []
    if steps:
        heading("Recommended solution")
        for index, step in enumerate(steps, start=1):
            pdf.multi_cell(0, 7, _safe(f"{index}. {step}"))
    elif result.get("solution_text"):
        heading("Recommended solution")
        pdf.multi_cell(0, 7, _safe(result["solution_text"]))

    # --- Alternatives -----------------------------------------------------
    alternatives = result.get("alternatives") or []
    if len(alternatives) > 1:
        heading("Other components worth checking")
        for item in alternatives:
            item_confidence = item.get("confidence", 0) or 0
            pdf.multi_cell(
                0,
                7,
                _safe(f"- {item.get('fault', '-')} ({item_confidence * 100:.0f}%)"),
            )

    # --- Disclaimer -------------------------------------------------------
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0,
        5,
        _safe(
            "Disclaimer: This is an AI-generated estimate based on the symptoms "
            "provided. Always confirm the actual fault with a physical hardware "
            "check or a qualified technician before replacing any part."
        ),
    )

    return bytes(pdf.output())
