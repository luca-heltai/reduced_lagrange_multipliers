"""Check every generated HTML page for unrendered or invalid mathematics.

Run after scripts/build_doc.sh with Playwright and its Chromium browser installed:
    python3 scripts/check_doc_math.py
"""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site", type=Path,
        default=Path(__file__).resolve().parents[1] / "build/docs/site",
    )
    parser.add_argument("--browser-executable", help="Use an installed Chromium binary")
    args = parser.parse_args()
    site = args.site.resolve()
    pages = sorted(site.rglob("*.html"))
    if not pages:
        parser.error(f"No HTML pages found in {site}; build the documentation first")

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(QuietHandler, directory=str(site)),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    failures = []
    formulas = 0
    try:
        with sync_playwright() as playwright:
            options = {}
            if args.browser_executable:
                options["executable_path"] = args.browser_executable
            browser = playwright.chromium.launch(**options)
            page = browser.new_page()
            for html in pages:
                relative = html.relative_to(site).as_posix()
                try:
                    response = page.goto(f"http://127.0.0.1:{server.server_port}/{relative}")
                    assert response.ok, f"HTTP {response.status}"
                    count = page.locator(".math").count()
                    formulas += count
                    if count:
                        page.wait_for_function("window.MathJax?.startup?.promise")
                        page.evaluate("async () => { await MathJax.startup.promise; }")
                        missing = page.locator(".math").evaluate_all(
                            "nodes => nodes.filter(n => !n.querySelector('mjx-container')).length"
                        )
                        assert not missing, f"{missing} formulas were not typeset"
                    errors = page.locator("mjx-merror, [data-mjx-error]").all_text_contents()
                    errors += page.locator('mtext[mathcolor="red"]').evaluate_all(
                        r"nodes => nodes.map(n => n.textContent).filter(t => /^\\[A-Za-z]+$/.test(t))"
                    )
                    assert not errors, f"MathJax errors: {errors}"
                    # MyST can consume unsupported delimiters before MathJax sees
                    # them. Check prose too, excluding intentional code examples.
                    raw = page.evaluate(r"""() => {
                        const main = document.querySelector('[role="main"]')
                            || document.querySelector('main') || document.body;
                        const prose = main.cloneNode(true);
                        prose.querySelectorAll('pre, code, mjx-container, script, style')
                            .forEach(node => node.remove());
                        return prose.textContent.match(
                            /\$[^$\n]+\$|\\(?:frac|int|partial|begin|lambda|cdot|nabla|mathrm|sqrt|Omega|Gamma)\b[^\n]*/g
                        ) || [];
                    }""")
                    assert not raw, f"Unrendered math in prose: {raw}"
                except Exception as error:
                    failures.append(f"{relative}: {error}")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    print(f"Checked {len(pages)} HTML pages and {formulas} formulas")
    if failures:
        raise SystemExit("\n".join(failures))
    print("All formulas rendered without MathJax errors or raw LaTeX in prose")


if __name__ == "__main__":
    main()
