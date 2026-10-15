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
from test_history import attempt, report, CONFIG
from test_particle_penalty import penalty_raw
from history import normalize

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
                real_mpi = [r for r in downloaded.json()['rows'] if r['case'].startswith('particle_penalty_')
                            and r['ranks'] == 2 and r['status'] == 'passed' and r['attempt_status'] == 'passed']
                if real_mpi:
                    page.goto(url)
                    page.wait_for_selector('#performance-results tbody tr')
                    page.get_by_label('case', exact=True).select_option(real_mpi[0]['case'])
                    page.get_by_label('ranks', exact=True).select_option('2')
                    assert '2 ranks' in page.locator('#performance-results').inner_text()
                    assert page.locator('#performance-chart svg').count() == 1
                empty = report({"source_sha": None, "baseline": None, "commits": []}, [])
                page.route("**/performance/data/history.json", lambda route: route.fulfill(json=empty))
                page.goto(url)
                page.wait_for_function("!document.querySelector('#performance-status').textContent.includes('Loading')")
                assert "No measurements yet" in page.locator("#performance-status").inner_text()
                assert page.request.get(url.replace("index.html", "data/history.json")).ok
                a, b = attempt(), attempt("b", "2", "failed")
                a["records"][0]["raw"]["extra"] = '<img src=x onerror="alert(1)">'
                penalty_case = "particle_penalty_1d_in_3d_vector"
                a['configuration'] = {**CONFIG, 'particle_penalty_ranks': [1, 2]}
                a["records"].append(normalize(penalty_raw(penalty_case), a['configuration']))
                a['records'].append(normalize({**penalty_raw(penalty_case), 'ranks': 2,
                    'active_particle_ranks': 2}, a['configuration']))
                a['raw_files'] = ['particle_penalty_benchmark-mpi2.jsonl']
                a["benchmark_sources"] = {"weak_term_assembly": "synthetic-blob",
                    "particle_penalty": "synthetic-penalty", "particle_penalty_native": "synthetic-manual",
                    "particle_penalty_coupling": "synthetic-coupling"}
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
                page.get_by_label("case", exact=True).select_option(penalty_case)
                assert penalty_case in page.locator("#performance-results").inner_text()
                assert "2.0000" in page.locator("#performance-results").inner_text()
                assert penalty_case in page.locator("#performance-series").inner_text()
                assert page.locator("#performance-chart svg").count() == 1
                page.get_by_label('ranks', exact=True).select_option('2')
                assert '2 ranks' in page.locator('#performance-results').inner_text()
                assert page.locator('#performance-results tbody tr').count() == 2
                assert '2 ranks' in page.locator('#performance-series').inner_text()
                assert page.locator('#performance-chart svg').count() == 1
                assert page.get_by_role('link', name='particle_penalty_benchmark-mpi2.jsonl', exact=True, include_hidden=True).count() == 1
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
