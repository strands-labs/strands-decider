"""Synthetic business documents (our own renders, no third-party data).

Invoices, receipts, purchase orders, forms with checkboxes and signatures, and
multi-year financial statements, drawn with PIL in varied layouts and fonts, some
with a "scanned" look (slight rotation, blur, grey paper, JPEG). Every answer is
computed from the fields that were drawn. Questions: field reading (choice among
same-type values), checks (yes/no: signed? box ticked? total above X? due before Y?),
and arithmetic over a table (change/sum between rows or years, FinQA-style options).
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import random

from common import row, write, yesno

FONTS = sorted(glob.glob("/usr/share/fonts/truetype/dejavu/*.ttf") + glob.glob("/usr/share/fonts/truetype/liberation/*.ttf"))
COMPANIES = ["Northwind Supply Co.", "Bluepeak Logistics", "Harbor & Finch LLP", "Quantum Parts Ltd.",
             "Greenleaf Foods", "Atlas Office Systems", "Redwood Medical", "Silverline Electric",
             "Orchid Textiles", "Pioneer Freight", "Cobalt Software Inc.", "Maple Street Bakery",
             "Summit Hardware", "Ironclad Security", "Lumen Labs", "Riverbend Dental"]
PEOPLE = ["J. Alvarez", "M. Chen", "S. Okafor", "R. Patel", "L. Novak", "A. Haddad", "K. Mensah", "D. Rossi",
          "E. Lindqvist", "T. Nakamura", "P. Dubois", "C. Moreau"]
ITEMS = ["Copy paper (box)", "Toner cartridge", "USB-C cable", "Desk lamp", "Safety gloves (pair)",
         "Pallet wrap", "Shipping labels", "Office chair", "Monitor stand", "Hand sanitizer", "Ladder 6ft",
         "Fire extinguisher", "Hard hat", "Cleaning service", "Consulting hours", "Freight charge",
         "Label printer", "Barcode scanner", "Storage bins", "First aid kit"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LINES = ["Net revenue", "Operating expenses", "Cost of sales", "Operating income", "Interest expense",
         "Net income", "Total assets", "Long-term debt", "Cash and equivalents", "Capital expenditures",
         "Depreciation", "Income tax expense"]


def pick_font(rng, size, bold=False):
    from PIL import ImageFont
    pool = [f for f in FONTS if ("Bold" in f) == bold] or FONTS
    return ImageFont.truetype(rng.choice(pool), size) if pool else ImageFont.load_default()


def money(v: float) -> str:
    return f"${v:,.2f}"


def date(rng):
    return rng.randint(1, 12), rng.randint(1, 28), rng.choice([2021, 2022, 2023, 2024])


def fmt_date(m, d, y, style):
    return [f"{m:02d}/{d:02d}/{y}", f"{MONTHS[m - 1]} {d}, {y}", f"{y}-{m:02d}-{d:02d}"][style]


def scanned(img, rng):
    from PIL import Image, ImageFilter
    if rng.random() < 0.4:
        img = img.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=(235, 235, 230))
        if rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
        img = Image.blend(img, Image.new("RGB", img.size, (225, 222, 210)), rng.uniform(0.05, 0.2))
    return img


def opts_from(gold: str, pool: list[str], rng, k=4):
    others = [p for p in dict.fromkeys(pool) if p != gold]
    rng.shuffle(others)
    vals = [*others[: k - 1], gold]
    if len(vals) < 3:
        return None, -1
    rng.shuffle(vals)
    return [[chr(65 + i), v] for i, v in enumerate(vals)], vals.index(gold)


def num_opts(v: float, rng, money_fmt=True):
    f = money if money_fmt else (lambda x: f"{x:,.1f}")
    # Gold's rank among the options is uniform: distractors all above, all below, or mixed.
    lo_, hi_ = [0.5, 0.6, 0.75, 0.8, 0.85, 0.9], [1.1, 1.15, 1.2, 1.25, 1.4, 1.5]
    n_lo = rng.randint(0, 3)
    cands = [v] + [v * m for m in rng.sample(lo_, n_lo) + rng.sample(hi_, 3 - n_lo)]
    vals = [f(x) for x in cands]
    if len(set(vals)) < 4:
        return None, -1
    order = sorted(range(4), key=lambda i: cands[i])
    return [[chr(65 + i), vals[j]] for i, j in enumerate(order)], order.index(0)


def invoice(idx, rng, out):
    from PIL import Image, ImageDraw
    W, H = rng.choice([(850, 1100), (900, 1150), (1000, 1300)])
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    kind = rng.choice(["INVOICE", "PURCHASE ORDER", "RECEIPT", "QUOTE"])
    vendor, client = rng.sample(COMPANIES, 2)
    num = f"{rng.choice(['INV', 'PO', 'R', 'Q'])}-{rng.randint(10000, 99999)}"
    im_, id_, iy = date(rng)
    dm, dd, dy = date(rng)
    dstyle = rng.randrange(3)
    x0 = rng.randint(40, 80)
    y = rng.randint(40, 70)
    dr.text((x0, y), vendor, fill=(20, 20, 60), font=pick_font(rng, rng.randint(24, 32), True))
    dr.text((W - 330, y), kind, fill=(90, 90, 90), font=pick_font(rng, rng.randint(26, 34), True))
    f = pick_font(rng, rng.randint(15, 18))
    y += 70
    dr.text((x0, y), f"{rng.randint(10, 999)} Market Street, Suite {rng.randint(1, 40)}", fill=0, font=f)
    dr.text((W - 330, y), f"No.: {num}", fill=0, font=f)
    y += 28
    dr.text((W - 330, y), f"Date: {fmt_date(im_, id_, iy, dstyle)}", fill=0, font=f)
    has_due = kind in ("INVOICE", "QUOTE")
    if has_due:
        dr.text((W - 330, y + 28), f"{'Due' if kind == 'INVOICE' else 'Valid until'}: {fmt_date(dm, dd, dy, dstyle)}",
                fill=0, font=f)
    y += 80
    dr.text((x0, y), "Bill to:" if kind != "RECEIPT" else "Customer:", fill=(90, 90, 90), font=f)
    dr.text((x0, y + 24), client, fill=0, font=pick_font(rng, 18, True))
    y += 90
    n = rng.randint(3, 7)
    items = rng.sample(ITEMS, n)
    qty = [rng.randint(1, 12) for _ in items]
    price = [round(rng.uniform(4, 400), 2) for _ in items]
    amt = [q * p for q, p in zip(qty, price, strict=True)]
    cols = [x0, x0 + int(W * 0.45), x0 + int(W * 0.6), x0 + int(W * 0.75)]
    dr.rectangle([x0 - 5, y - 4, W - x0 + 5, y + 26], fill=rng.choice([(230, 236, 245), (240, 240, 240), (225, 245, 230)]))
    for c, h in zip(cols, ["Description", "Qty", "Unit price", "Amount"], strict=True):
        dr.text((c, y), h, fill=0, font=pick_font(rng, 16, True))
    y += 36
    for it, q, p, a in zip(items, qty, price, amt, strict=True):
        for c, v in zip(cols, [it, str(q), money(p), money(a)], strict=True):
            dr.text((c, y), v, fill=0, font=f)
        y += 30
        if rng.random() < 0.5:
            dr.line([x0, y - 4, W - x0, y - 4], fill=(210, 210, 210))
    sub = sum(amt)
    taxr = rng.choice([0, 0.05, 0.07, 0.08, 0.1])
    tax = round(sub * taxr, 2)
    total = sub + tax
    y += 20
    for lab, v in [("Subtotal", money(sub)), (f"Tax ({int(taxr * 100)}%)", money(tax)), ("TOTAL", money(total))]:
        dr.text((cols[2], y), lab, fill=0, font=pick_font(rng, 17, lab == "TOTAL"))
        dr.text((cols[3], y), v, fill=0, font=pick_font(rng, 17, lab == "TOTAL"))
        y += 30
    paid = rng.random() < 0.4
    if paid:
        st = pick_font(rng, 44, True)
        dr.text((x0 + 60, y + 40), "PAID", fill=(200, 30, 30), font=st)
    signed = rng.random() < 0.5
    y = H - 160
    dr.text((x0, y), "Authorized signature:", fill=(90, 90, 90), font=f)
    dr.line([x0 + 200, y + 22, x0 + 480, y + 22], fill=0)
    if signed:
        pts = [(x0 + 210 + i * 12, y + 14 + rng.randint(-8, 6)) for i in range(20)]
        dr.line(pts, fill=(20, 20, 120), width=2)
    img = scanned(img, rng)
    name = f"docs/d{idx:06d}.jpg"
    img.save(os.path.join(out, name), quality=rng.randint(70, 92))
    kw = dict(source="docs", images=[name], source_id=f"doc-{idx}")
    rows = []
    o, g = num_opts(total, rng)
    if o:
        rows.append(row("choice", rng.choice([f"What is the total amount on this {kind.lower()}?",
                                              "What is the grand total?", "How much is due in total?"]),
                        o, g, task="doc/total", **kw))
    o, g = opts_from(vendor, COMPANIES, rng)
    rows.append(row("choice", f"Which company issued this {kind.lower()}?", o, g, task="doc/vendor", **kw))
    i = rng.randrange(n)
    o, g = opts_from(str(qty[i]), [str(q) for q in range(1, 13)], rng)
    rows.append(row("choice", f"What quantity of '{items[i]}' is listed?", o, g, task="doc/qty", **kw))
    thr = rng.choice([100, 250, 500, 1000, 2000, 5000])
    if abs(total - thr) > 0.05 * thr:
        rows.append(yesno(f"Is the total above {money(thr)}?", total > thr, rng, task="doc/total_check", **kw))
    rows.append(yesno(f"Is this {kind.lower()} signed?", signed, rng, task="doc/signed", **kw))
    if kind in ("INVOICE", "RECEIPT"):
        rows.append(yesno(f"Is this {kind.lower()} marked as paid?", paid, rng, task="doc/paid", **kw))
    if rng.random() < 0.5:
        it = rng.choice(items) if rng.random() < 0.5 else rng.choice([x for x in ITEMS if x not in items])
        rows.append(yesno(f"Does this {kind.lower()} include '{it}'?", it in items, rng, task="doc/item", **kw))
    o, g = opts_from(num, [f"{num[:num.index('-') + 1]}{rng.randint(10000, 99999)}" for _ in range(5)], rng)
    if o:
        rows.append(row("choice", f"What is the {kind.lower()} number?", o, g, task="doc/number", **kw))
    if taxr > 0:
        o, g = num_opts(tax, rng)
        if o:
            rows.append(row("choice", "How much tax is charged?", o, g, task="doc/tax", **kw))
    return rows


def form(idx, rng, out):
    from PIL import Image, ImageDraw
    W, H = 850, 1100
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    title = rng.choice(["Equipment Inspection Form", "Incident Report", "Vendor Registration Form",
                        "Leave Request", "Delivery Acceptance Form", "Site Safety Checklist"])
    dr.text((60, 50), title, fill=0, font=pick_font(rng, 30, True))
    f = pick_font(rng, 18)
    y = 130
    fields = {"Name": rng.choice(PEOPLE), "Employee ID": str(rng.randint(100000, 999999)),
              "Department": rng.choice(["Warehouse", "Finance", "Facilities", "IT", "Logistics", "Lab"]),
              "Date": fmt_date(*date(rng), rng.randrange(3))}
    for k, v in fields.items():
        dr.text((60, y), f"{k}:", fill=(80, 80, 80), font=f)
        dr.text((260, y), v, fill=(10, 10, 90), font=pick_font(rng, 19))
        dr.line([255, y + 24, 700, y + 24], fill=(150, 150, 150))
        y += 50
    y += 20
    checks = rng.sample(["Hard hat worn", "Fire exit clear", "Spill kit available", "Guard rails in place",
                         "Lockout/tagout applied", "First aid kit stocked", "Goods undamaged",
                         "Quantity matches PO", "Manager approval attached", "Eye protection worn"], 5)
    ticked = [rng.random() < 0.5 for _ in checks]
    for c, t in zip(checks, ticked, strict=True):
        dr.rectangle([60, y, 80, y + 20], outline=0, width=2)
        if t:
            dr.line([63, y + 10, 70, y + 17, 78, y + 3], fill=(0, 0, 0), width=3)
        dr.text((95, y - 1), c, fill=0, font=f)
        y += 40
    signed = rng.random() < 0.5
    dr.text((60, H - 150), "Signature:", fill=(80, 80, 80), font=f)
    dr.line([170, H - 128, 470, H - 128], fill=0)
    if signed:
        pts = [(180 + i * 13, H - 136 + rng.randint(-8, 6)) for i in range(20)]
        dr.line(pts, fill=(20, 20, 120), width=2)
    img = scanned(img, rng)
    name = f"docs/f{idx:06d}.jpg"
    img.save(os.path.join(out, name), quality=rng.randint(70, 92))
    kw = dict(source="docs", images=[name], source_id=f"form-{idx}")
    rows = []
    for _ in range(2):
        i = rng.randrange(5)
        rows.append(yesno(f"Is the box '{checks[i]}' checked on this form?", ticked[i], rng, task="doc/checkbox", **kw))
    rows.append(yesno("Has the form been signed?", signed, rng, task="doc/signed", **kw))
    o, g = opts_from(fields["Name"], PEOPLE, rng)
    rows.append(row("choice", "Whose name is on the form?", o, g, task="doc/field", **kw))
    o, g = opts_from(fields["Department"], ["Warehouse", "Finance", "Facilities", "IT", "Logistics", "Lab"], rng)
    rows.append(row("choice", "Which department is entered?", o, g, task="doc/field", **kw))
    cnt = sum(ticked)
    rows.append(row("choice", "How many boxes are checked?", [[str(k), ""] for k in range(6)], cnt, task="doc/count", **kw))
    return rows


def statement(idx, rng, out):
    from PIL import Image, ImageDraw
    W, H = rng.choice([(900, 600), (1000, 700), (800, 520)])
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    co = rng.choice(COMPANIES)
    y0 = rng.randint(2012, 2021)
    years = [y0, y0 + 1, y0 + 2][: rng.choice([2, 3])]
    if rng.random() < 0.5:
        years = years[::-1]
    lines = rng.sample(LINES, rng.randint(4, 7))
    unit = rng.choice(["(in millions)", "(in thousands)", "($ in millions)"])
    vals = {name: [round(rng.uniform(20, 3000), 1) for _ in years] for name in lines}
    dr.text((40, 30), f"{co} - Selected financial data {unit}", fill=0, font=pick_font(rng, 22, True))
    f = pick_font(rng, 17)
    y = 90
    cx = [40] + [int(W * 0.5) + i * int(W * 0.16) for i in range(len(years))]
    for c, h in zip(cx, ["", *[str(v) for v in years]], strict=True):
        dr.text((c, y), h, fill=0, font=pick_font(rng, 17, True))
    dr.line([40, y + 26, W - 40, y + 26], fill=0)
    y += 36
    for name in lines:
        dr.text((cx[0], y), name, fill=0, font=f)
        for c, v in zip(cx[1:], vals[name], strict=True):
            dr.text((c, y), f"{v:,.1f}", fill=0, font=f)
        y += 32
    img = scanned(img, rng)
    name = f"docs/s{idx:06d}.jpg"
    img.save(os.path.join(out, name), quality=rng.randint(75, 95))
    kw = dict(source="docs", images=[name], source_id=f"stmt-{idx}")
    rows = []
    item = rng.choice(lines)
    i, j = rng.sample(range(len(years)), 2)
    a_, b_ = sorted([i, j], key=lambda k: years[k])
    ch = vals[item][b_] - vals[item][a_]
    o, g = num_opts(ch, rng, money_fmt=False) if abs(ch) > 1 else (None, -1)
    if o:
        rows.append(row("choice", f"What is the change in {item.lower()} from {years[a_]} to {years[b_]}?", o, g,
                        task="doc/fin_change", **kw))
    l2 = rng.choice(lines)
    k = rng.randrange(len(years))
    o, g = num_opts(vals[l2][k], rng, money_fmt=False)
    if o:
        rows.append(row("choice", f"What was {l2.lower()} in {years[k]}?", o, g, task="doc/fin_value", **kw))
    if abs(ch) > 0.02 * max(vals[item]):
        rows.append(yesno(f"Did {item.lower()} increase from {years[a_]} to {years[b_]}?", ch > 0, rng,
                          task="doc/fin_trend", **kw))
    l3, l4 = rng.sample(lines, 2)
    k = rng.randrange(len(years))
    s = vals[l3][k] + vals[l4][k]
    o, g = num_opts(s, rng, money_fmt=False)
    if o and rng.random() < 0.5:
        rows.append(row("choice", f"What is the sum of {l3.lower()} and {l4.lower()} in {years[k]}?", o, g,
                        task="doc/fin_sum", **kw))
    return rows


def make(args):
    idx, out, seed = args
    rng = random.Random(seed)
    r = rng.random()
    if r < 0.5:
        return invoice(idx, rng, out)
    if r < 0.75:
        return statement(idx, rng, out)
    return form(idx, rng, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=1300)
    ap.add_argument("--max-rows-per-doc", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "docs"), exist_ok=True)
    jobs = [(i, a.out, a.seed * 1_000_003 + i) for i in range(a.n_docs)]
    rng = random.Random(a.seed)
    rows = []
    with mp.Pool(a.workers) as pool:
        for part in pool.imap(make, jobs, chunksize=8):
            part = [p for p in part if p["options"]]
            rng.shuffle(part)
            rows += part[: a.max_rows_per_doc]
    write(os.path.join(a.out, "docs.jsonl"), rows)


if __name__ == "__main__":
    main()
