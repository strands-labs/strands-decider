"""Local preview server for the docs/ tree rendered from docs/site-manifest.json.

The strands-agents/harness-sdk site is the source of truth for how these pages
actually look; this script is a developer convenience for sanity-checking the
manifest wiring, nav grouping, and markdown content without pushing. Markdown
is rendered client-side with marked.js so the server has no non-stdlib deps.

Run: python scripts/preview_docs.py [--port 8765] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "docs" / "site-manifest.json"


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{product_name} — docs preview</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css@5.6.1/github-markdown-light.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.10.0/styles/github.min.css">
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #24292f; }}
  .layout {{ display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }}
  nav {{ border-right: 1px solid #d0d7de; padding: 24px 16px; background: #f6f8fa; overflow-y: auto; position: sticky; top: 0; height: 100vh; }}
  nav h1 {{ font-size: 14px; margin: 0 0 16px; text-transform: uppercase; letter-spacing: .05em; color: #57606a; }}
  nav .group {{ font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: #57606a; margin: 18px 0 6px; }}
  nav a {{ display: block; padding: 6px 8px; border-radius: 6px; color: #24292f; text-decoration: none; font-size: 14px; }}
  nav a:hover {{ background: #eaeef2; }}
  nav a.active {{ background: #0969da; color: #fff; }}
  main {{ padding: 32px 48px; max-width: 920px; }}
  main .page-meta {{ color: #57606a; font-size: 14px; margin-bottom: 24px; }}
  .banner {{ background: #fff8c5; border: 1px solid #d4a72c; padding: 8px 12px; font-size: 13px; color: #633c01; border-radius: 6px; margin-bottom: 20px; }}
  .markdown-body {{ box-sizing: border-box; }}
  code {{ background: rgba(175,184,193,.2); padding: .2em .4em; border-radius: 6px; }}
  pre code {{ background: none; padding: 0; }}
</style>
</head>
<body>
<div class="layout">
  <nav>
    <h1>{product_name}</h1>
    {nav_html}
  </nav>
  <main>
    <div class="banner">Local preview of <code>docs/</code> via <code>docs/site-manifest.json</code>. The production site (strands-agents/harness-sdk) renders these pages with its own styling.</div>
    <div class="page-meta" id="page-meta"></div>
    <article id="content" class="markdown-body">Loading…</article>
  </main>
</div>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11.10.0/lib/common.min.js"></script>
<script>
const PAGES = {pages_json};
const BY_SLUG = Object.fromEntries(PAGES.map(p => [p.slug, p]));

marked.setOptions({{
  highlight: (code, lang) => {{
    try {{ return lang && hljs.getLanguage(lang) ? hljs.highlight(code, {{ language: lang }}).value : hljs.highlightAuto(code).value; }}
    catch (e) {{ return code; }}
  }},
  gfm: true,
  breaks: false,
}});

async function render(slug) {{
  const page = BY_SLUG[slug];
  if (!page) {{ document.getElementById('content').textContent = 'Unknown page: ' + slug; return; }}
  document.querySelectorAll('nav a').forEach(a => a.classList.toggle('active', a.dataset.slug === slug));
  document.title = page.title + ' — {product_name}';
  document.getElementById('page-meta').textContent = page.description || '';
  const resp = await fetch('/raw/' + encodeURIComponent(page.source));
  if (!resp.ok) {{
    document.getElementById('content').textContent = 'Failed to load ' + page.source + ' (' + resp.status + ')';
    return;
  }}
  const md = await resp.text();
  document.getElementById('content').innerHTML = marked.parse(md);
}}

function currentSlug() {{
  const h = location.hash.replace(/^#\\/?/, '');
  return h || PAGES[0].slug;
}}
window.addEventListener('hashchange', () => render(currentSlug()));
document.querySelectorAll('nav a').forEach(a => a.addEventListener('click', e => {{
  e.preventDefault();
  location.hash = '#/' + a.dataset.slug;
}}));
render(currentSlug());
</script>
</body>
</html>
"""


def build_nav(pages: list[dict]) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        groups[p.get("nav", {}).get("group", "Pages")].append(p)
    for group_pages in groups.values():
        group_pages.sort(key=lambda p: p.get("nav", {}).get("order", 0))
    parts: list[str] = []
    for group, group_pages in groups.items():
        parts.append(f'<div class="group">{group}</div>')
        for p in group_pages:
            parts.append(f'<a data-slug="{p["slug"]}" href="#/{p["slug"]}">{p["title"]}</a>')
    return "\n".join(parts)


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        print(f"error: {MANIFEST_PATH} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(MANIFEST_PATH.read_text())


class Handler(BaseHTTPRequestHandler):
    manifest: ClassVar[dict] = {}

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            product = self.manifest.get("product", {})
            html = INDEX_HTML.format(
                product_name=product.get("name", product.get("id", "Docs")),
                nav_html=build_nav(self.manifest["pages"]),
                pages_json=json.dumps(self.manifest["pages"]),
            )
            self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/manifest.json":
            self._send(
                HTTPStatus.OK,
                json.dumps(self.manifest, indent=2).encode("utf-8"),
                "application/json",
            )
            return
        if path.startswith("/raw/"):
            rel = path[len("/raw/") :]
            # Resolve inside repo root and reject escapes.
            target = (REPO_ROOT / rel).resolve()
            if REPO_ROOT not in target.parents and target != REPO_ROOT:
                self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
                return
            if not target.is_file() or target.suffix.lower() != ".md":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            self._send(
                HTTPStatus.OK,
                target.read_bytes(),
                "text/markdown; charset=utf-8",
            )
            return
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def log_message(self, fmt: str, *args) -> None:  # quieter default output
        sys.stderr.write(f"[preview] {self.address_string()} - {fmt % args}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    Handler.manifest = load_manifest()
    # Validate sources exist up front so typos in the manifest surface immediately.
    missing = [
        p["source"] for p in Handler.manifest["pages"] if not (REPO_ROOT / p["source"]).is_file()
    ]
    if missing:
        print("error: manifest references missing files: " + ", ".join(missing), file=sys.stderr)
        sys.exit(1)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving docs preview at {url} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
