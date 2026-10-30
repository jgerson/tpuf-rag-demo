"""
Builds samples/harbor-lane-handbook.pdf, the fictional document the demo uses.

Hobby project note: "Harbor Lane Coffee Roasters" is made up, and so is every
policy in this handbook. I generate the PDF myself so the repo doesn't ship
anyone else's copyrighted document.

The PDF is deliberately a little messy, the way real PDFs are:
  - a running header and footer on every page (noise a parser should drop)
  - a two-column page (raw text extraction reads it in the wrong order)
  - a table (raw extraction flattens it into a jumble of numbers)
  - one section where the topic changes mid-way, and two sections that flow
    into each other, so recursive and semantic chunking cut differently

Run:  python scripts/make_sample_pdf.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

OUT = Path(__file__).resolve().parent.parent / "samples" / "harbor-lane-handbook.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5, spaceBefore=6, spaceAfter=2)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=13.5, spaceAfter=6)


def h1(t): return Paragraph(t, H1)
def h2(t): return Paragraph(t, H2)
def h3(t): return Paragraph(t, H3)
def p(t): return Paragraph(t, BODY)


def header_footer(canvas, doc):
    # Running header and footer: the kind of noise that ends up in every chunk
    # if the parser doesn't recognise and drop it.
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(inch, LETTER[1] - 0.6 * inch, "Harbor Lane Coffee Roasters  |  Employee Handbook  |  INTERNAL")
    canvas.drawRightString(LETTER[0] - inch, 0.55 * inch, f"Page {doc.page}  |  v3.2  |  Fictional sample document")
    canvas.restoreState()


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUT), pagesize=LETTER, title="Harbor Lane Employee Handbook (fictional)",
                          author="tpuf-rag-demo hobby project")
    w, h = LETTER
    one = Frame(inch, inch, w - 2 * inch, h - 2 * inch, id="one")
    gap = 0.3 * inch
    col_w = (w - 2 * inch - gap) / 2
    left = Frame(inch, inch, col_w, h - 2 * inch, id="left")
    right = Frame(inch + col_w + gap, inch, col_w, h - 2 * inch, id="right")
    doc.addPageTemplates([
        PageTemplate(id="OneCol", frames=[one], onPage=header_footer),
        PageTemplate(id="TwoCol", frames=[left, right], onPage=header_footer),
    ])

    s = []
    s += [h1("Harbor Lane Employee Handbook"),
          p("This handbook explains how we work together at Harbor Lane Coffee Roasters. It covers time off, "
            "expenses, travel, security and benefits. Harbor Lane is a fictional company and this handbook is a "
            "sample document for a hobby project."),
          p("If something here conflicts with your offer letter, your offer letter wins. Questions go to "
            "people-team@harborlane.example.")]

    s += [h2("1. Time off"),
          h3("1.1 Vacation"),
          p("Full-time employees receive 20 days of paid vacation each calendar year. Part-time employees accrue "
            "vacation in proportion to their scheduled hours."),
          p("Unused vacation can roll over into the next calendar year, up to a maximum of 5 days. Anything above "
            "5 days expires on January 1. We would rather you rest than bank days."),
          p("Ask your manager for approval at least two weeks ahead for trips longer than three days. During the "
            "holiday roasting season, from November 15 to December 31, we may limit how many people on the roastery "
            "floor are out at once."),
          h3("1.2 Sick days"),
          p("Take the time you need when you are unwell. You do not need a doctor's note for absences of three "
            "consecutive days or fewer. Tell your manager as early as you can so shifts can be covered."),
          h3("1.3 Parental leave"),
          p("All employees who have completed 90 days of service are eligible for paid parental leave. The amount "
            "depends on your role and is set out in the table below. Leave can be taken in one block or split into "
            "two blocks within the first year."),
          ]

    table = Table([
        ["Leave type", "Roastery and cafe staff", "Office staff", "Pay"],
        ["Primary caregiver", "16 weeks", "16 weeks", "100% of base pay"],
        ["Secondary caregiver", "6 weeks", "8 weeks", "100% of base pay"],
        ["Adoption or foster placement", "12 weeks", "12 weeks", "100% of base pay"],
        ["Bereavement (immediate family)", "5 days", "5 days", "100% of base pay"],
    ], colWidths=[1.9 * inch, 1.6 * inch, 1.2 * inch, 1.6 * inch])
    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
    ]))
    s += [table, Spacer(1, 10),
          p("Parental leave runs alongside any leave provided by law, not on top of it.")]

    # Page 2: two columns. Raw text extraction tends to interleave or misorder these.
    s += [NextPageTemplate("TwoCol"), PageBreak()]
    s += [h2("2. Expenses"),
          h3("2.1 What you can expense"),
          p("You can expense reasonable costs you incur while doing your job: travel, meals with customers or "
            "suppliers, training courses and small equipment. Keep receipts for anything over $25."),
          h3("2.2 Approvals"),
          p("Expenses under $500 need approval from your manager. Expenses of $500 or more also need approval from "
            "the finance lead, Dana Okafor. Equipment over $1,500 must be bought through the finance team rather "
            "than expensed."),
          p("Submit expenses within 30 days. Claims older than 90 days will not be reimbursed."),
          # This paragraph deliberately drifts to a different topic mid-section.
          p("A note on the office kitchen while we're here: the espresso machine on the second floor is for "
            "everyone, and it is descaled every Friday afternoon. Please don't leave cups in the sink, and report "
            "broken equipment to facilities@harborlane.example."),
          FrameBreak(),
          h2("3. Travel"),
          h3("3.1 Booking"),
          p("Book flights and hotels through the TravelDesk portal so we can find you if something goes wrong. "
            "Economy class for flights under six hours; premium economy is fine for longer flights."),
          h3("3.2 Meals and per diem"),
          p("Domestic trips: up to $60 per day for meals. International trips: up to the equivalent of $85 per "
            "day, or GBP 70 per day in the United Kingdom. Alcohol is not covered unless you are hosting a customer."),
          p("Travel days also count toward the per diem. If a trip ends before noon, claim half a day."),
          ]

    s += [NextPageTemplate("OneCol"), PageBreak()]
    s += [h2("4. Security"),
          h3("4.1 Laptops and phones"),
          p("Company laptops are encrypted and managed centrally. Never disable disk encryption or the endpoint "
            "agent. Lock your screen when you step away, even in the office."),
          p("If your laptop or phone is lost or stolen, report it to security@harborlane.example within one hour. "
            "We can wipe the device remotely, and the sooner we know, the less there is to worry about. You will "
            "not get in trouble for reporting a missing device."),
          h3("4.2 Passwords and sign-in"),
          p("Use the company password manager for every work account and turn on single sign-on wherever it is "
            "offered. Policy HL-SEC-07 requires a hardware security key for anyone with access to payment systems."),
          h3("4.3 Reporting incidents"),
          p("If you think something is wrong, such as a strange email asking for gift cards or a supplier "
            "suddenly changing bank details, tell security before you act. Pausing to check is always the "
            "right call."),
          h2("5. Benefits"),
          p("Harbor Lane pays 90% of health insurance premiums for employees and 70% for dependants. We match "
            "retirement contributions up to 4% of base pay, and every employee gets a monthly coffee allowance "
            "of two bags from the current roast."),
          p("Learning budget: each employee has $1,000 per year for courses, books and conferences. Unused "
            "learning budget does not roll over."),
          ]
    doc.build(s)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
