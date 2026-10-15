"""Browser checks against a built Sphinx site. Synthetic data never leave tests.

Run: uv run --with playwright python scripts/performance/tests/viewer.py
Install Chromium first: uv run --with playwright python -m playwright install chromium
"""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile
import threading

from playwright.sync_api import sync_playwright
from test_history import attempt, report

ROOT = Path(__file__).resolve().parents[3]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # Exercise deployment below /immersx/, not only a server root.
        site = Path(tmp) / "immersx"
        shutil.copytree(ROOT / "build/docs/site", site)
        # Use the current asset even when iterating after the Sphinx build.
        shutil.copyfile(ROOT / "doc/_static/performance.js", site / "_static/performance.js")
        server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=tmp))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        url = f"http://127.0.0.1:{server.server_port}/immersx/performance/index.html"
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                downloaded = page.request.get(url.replace("index.html", "data/history.json"))
                assert downloaded.ok and downloaded.json()["schema_version"] == 1
                empty = report({"source_sha": None, "baseline": None, "commits": []}, [])
                page.route("**/performance/data/history.json", lambda route: route.fulfill(json=empty))
                page.goto(url)
                page.wait_for_function("!document.querySelector('#performance-status').textContent.includes('Loading')")
                assert "No measurements yet" in page.locator("#performance-status").inner_text()
                assert page.request.get(url.replace("index.html", "data/history.json")).ok
                a, b = attempt(), attempt("b", "2", "failed")
                a["records"][0]["raw"]["extra"] = '<img src=x onerror="alert(1)">'
                a["benchmark_sources"] = {"weak_term_assembly": "synthetic-blob"}
                # Keep an archived v1 record with no new timing-contract fields.
                del b["records"][0]["timing_contract"]
                data = report({"source_sha": "c", "baseline": "a", "commits": ["a", "b", "c"]}, [a, b])
                page.route("**/performance/data/history.json", lambda route: route.fulfill(json=data))
                page.goto(url)
                page.wait_for_selector("#performance-results tbody tr")
                assert "missing" in page.locator("#performance-coverage").inner_text()
                assert "failed" in page.locator("#performance-coverage").inner_text()
                assert "Failed attempt" in page.locator("#performance-results").inner_text()
                assert page.locator("#performance-chart svg").count() == 1
                assert page.locator("#performance-attempts img").count() == 0
                page.get_by_label("case", exact=True).select_option("mass_q1")
                assert "case=mass_q1" in page.url
                page.goto(url + "#attempt-a-1-1")
                page.wait_for_selector("#attempt-a-1-1[open]")
                assert "synthetic" in page.locator("#attempt-a-1-1").inner_text()
                assert "synthetic-blob" in page.locator("#attempt-a-1-1").inner_text()
                assert "legacy-unspecified" in page.locator("#attempt-a-1-1").inner_text()
                page.unroute("**/performance/data/history.json")
                page.route("**/performance/data/history.json", lambda route: route.fulfill(status=404, body="missing"))
                page.goto(url)
                page.wait_for_function("document.querySelector('#performance-status').textContent.includes('History unavailable')")
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown(); thread.join(); server.server_close()
    print("Viewer: empty, partial, failed, filtered, permalink, escaped and unavailable history passed under /immersx/")


if __name__ == "__main__":
    main()
