"""Draw a page of OCR boxes as SVG, for the walkthrough notebook and the README picture.

SVG keeps the Arabic text readable: the browser or GitHub shapes it and lays it out right to
left. Only the standard library is used.
"""

from __future__ import annotations

import textwrap
from collections.abc import Collection, Sequence
from html import escape
from typing import Any

Box = dict[str, Any]
# (from id, to id, colour, label): an arrow from the bottom of one box to the top of another
Link = tuple[str, str, str, str]

PALETTE = [  # one soft fill per group; colour-blind safe enough for nine groups plus grey
    "#cfe3f7",
    "#f9d9b6",
    "#d4ecd0",
    "#f3cfd9",
    "#e3d7f2",
    "#fbf0b5",
    "#cdeeee",
    "#e8dccb",
    "#dfe7c4",
]
GREY = "#e4e4e4"
GREEN, RED, DASHED = "#1a7f37", "#cf222e", "#8c959f"


def ascii_only(svg: str) -> str:
    """Write non-ASCII characters (Arabic, arrows) as XML character references.

    GitHub shows a notebook's SVG output without saying it is UTF-8, so raw Arabic would come
    out garbled there; character references read the same everywhere.
    """
    return svg.encode("ascii", "xmlcharrefreplace").decode("ascii")


def colours_for(groups: list[list[str]], skip: frozenset[str] = frozenset()) -> dict[str, str]:
    """Give every box in a group the same fill; boxes in `skip` (the header) stay grey."""
    out: dict[str, str] = {}
    k = 0
    for g in groups:
        if set(g) <= skip:
            out.update(dict.fromkeys(g, GREY))
            continue
        out.update(dict.fromkeys(g, PALETTE[k % len(PALETTE)]))
        k += 1
    return out


def page_svg(
    page: dict[str, Any],
    fill: dict[str, str],
    links: Sequence[Link] = (),
    gloss: dict[str, str] | None = None,
    title: str = "",
    flagged: Collection[str] = (),
    scale: float = 0.5,
) -> str:
    """Return an SVG picture of the page: one rectangle per box, then the arrows on top.

    Each box shows its id, an English gloss if given, and the first words of its text. Boxes
    in `flagged` get a red outline.
    """
    boxes = {b["id"]: b for b in page["boxes"]}
    width = page.get("page", {}).get("width") or max(b["bbox"][2] for b in boxes.values())
    height = page.get("page", {}).get("height") or max(b["bbox"][3] for b in boxes.values())
    top = 30 if title else 6
    w, h = width * scale + 12, height * scale + top + 6
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}" font-family="Segoe UI, Arial, sans-serif">',
        '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        'fill="context-stroke"/></marker></defs>',
        f'<rect width="{w:.0f}" height="{h:.0f}" fill="#ffffff"/>',
    ]
    if title:
        out.append(f'<text x="8" y="20" font-size="15" font-weight="600">{escape(title)}</text>')

    def xy(v: float, offset: float) -> float:
        return v * scale + offset

    for b in boxes.values():
        x0, y0, x1, y1 = b["bbox"]
        x, y, bw, bh = xy(x0, 6), xy(y0, top), (x1 - x0) * scale, (y1 - y0) * scale
        stroke = (
            f'stroke="{RED}" stroke-width="2.5"'
            if b["id"] in flagged
            else ('stroke="#57606a" stroke-width="0.8"')
        )
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" '
            f'fill="{fill.get(b["id"], "#ffffff")}" {stroke}/>'
        )
        label = b["id"] + (f" · {gloss[b['id']]}" if gloss and b["id"] in gloss else "")
        out.append(
            f'<text x="{x + 4:.1f}" y="{y + 13:.1f}" font-size="10.5" font-weight="600">'
            f"{escape(label)}</text>"
        )
        if bh > 34:  # room for a second line: the first words, right-aligned like the page
            words = textwrap.shorten(b["text"], int(bw / 6.5), placeholder="…")
            words = "\u2067" + words + "\u2069"  # isolate as right-to-left text (RLI ... PDI)
            out.append(
                f'<text x="{x + bw - 4:.1f}" y="{y + 28:.1f}" font-size="11" '
                f'text-anchor="end" fill="#24292f">{escape(words)}</text>'
            )

    for a, b, colour, label in links:
        (ax0, _, ax1, ay1), (bx0, by0, bx1, _) = boxes[a]["bbox"], boxes[b]["bbox"]
        sx, sy = xy((ax0 + ax1) / 2, 6), xy(ay1, top) - 2
        ex, ey = xy((bx0 + bx1) / 2, 6), xy(by0, top) + 2
        dash = ' stroke-dasharray="4 3"' if colour == DASHED else ""
        if abs(sx - ex) < 1:  # same column: a short straight arrow
            path = f"M{sx:.1f},{sy:.1f} L{ex:.1f},{ey:.1f}"
            lx, ly = sx + 6, (sy + ey) / 2 + 4
        else:  # column jump: down from A, up the gap between the columns, into the top of B
            left, right = (bx1, ax0) if bx1 <= ax0 else (ax1, bx0)
            gx = xy((left + right) / 2, 6)
            path = f"M{sx:.1f},{sy:.1f} V{sy + 8:.1f} H{gx:.1f} V{ey - 8:.1f} H{ex:.1f} V{ey:.1f}"
            lx, ly = gx + 4, (sy + ey) / 2
        out.append(
            f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2"{dash} '
            'marker-end="url(#head)"/>'
        )
        if label:
            out.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11" font-weight="700" '
                f'fill="{colour}" stroke="#ffffff" stroke-width="3" paint-order="stroke">'
                f"{escape(label)}</text>"
            )
    out.append("</svg>")
    return ascii_only("\n".join(out))


def bars_svg(
    rows: Sequence[tuple[str, Sequence[tuple[float, str]]]],
    legend: Sequence[tuple[str, str]] = (),
    threshold: float = 0.5,
    title: str = "",
) -> str:
    """Return a horizontal bar chart: one row per label, one bar per (value, colour) in it.

    A dashed line marks `threshold`, so you can see which side of it each value falls.
    """
    left, bar_w, bar_h, gap = 290, 360, 12, 10
    per_row = max((len(bars) for _, bars in rows), default=1) * (bar_h + 2) + gap
    top = 46 if title else 24
    h = top + per_row * len(rows) + 10
    w = left + bar_w + 60
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Segoe UI, Arial, sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
    ]
    if title:
        out.append(f'<text x="8" y="18" font-size="15" font-weight="600">{escape(title)}</text>')
    x = 8
    for name, colour in legend:
        out.append(f'<rect x="{x}" y="{top - 18}" width="10" height="10" fill="{colour}"/>')
        out.append(f'<text x="{x + 14}" y="{top - 9}">{escape(name)}</text>')
        x += 24 + 7 * len(name)
    for i, (label, bars) in enumerate(rows):
        y = top + i * per_row
        out.append(f'<text x="{left - 8}" y="{y + bar_h}" text-anchor="end">{escape(label)}</text>')
        for j, (value, colour) in enumerate(bars):
            by = y + j * (bar_h + 2)
            out.append(
                f'<rect x="{left}" y="{by}" width="{value * bar_w:.1f}" height="{bar_h}" '
                f'fill="{colour}"/>'
            )
            out.append(
                f'<text x="{left + value * bar_w + 4:.1f}" y="{by + bar_h - 2}" '
                f'font-size="11">{value:.2f}</text>'
            )
    tx = left + threshold * bar_w
    out.append(
        f'<line x1="{tx:.1f}" y1="{top - 4}" x2="{tx:.1f}" y2="{h - 6}" stroke="#57606a" '
        'stroke-dasharray="4 3"/>'
    )
    out.append(
        f'<text x="{tx + 3:.1f}" y="{h - 8}" font-size="10" fill="#57606a">{threshold}</text>'
    )
    out.append("</svg>")
    return ascii_only("\n".join(out))
