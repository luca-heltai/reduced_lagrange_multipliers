"""Synthetic fixtures here are tests, never measurements of repository SHAs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from history import (archive_attempt, attempts, normalize, parse_jsonl, pending,
                     report, selected_attempt, series_key)
from pipeline import ancestry, export, git, measure, merge, plan, save
from archive import checkout, validate_publication

CONFIG = {"repetitions": 9, "refinements": 3, "build_type": "Release",
          "quadrature_recipe": "synthetic test rule"}


def raw():
    return {"case": "mass_q1", "dimension": 2, "ranks": 1, "dofs": 81,
            "cells": 64, "quadrature_points_per_cell": 4, "repetitions": 9,
            "relative_matrix_error": 0, "warm_immersx_median_ms": 2.,
            "warm_immersx_mad_ms": 0.01, "warm_dealii_median_ms": 1.,
            "warm_dealii_mad_ms": 0.01, "warm_ratio": 2.}


def attempt(sha="a", run="1", status="passed"):
    return {"id": sha + "-" + run + "-1", "sha": sha, "run": run,
            "attempt": 1, "started_at": "2026-01-01T00:00:0" + run + "+00:00",
            "status": status, "harness_version": 1, "harness_sha": "test-only",
            "environment": {"cpu": "synthetic", "image": "fixture"},
            "configuration": CONFIG, "run_url": "https://github.com/example/test/actions/runs/1",
            "records": [normalize(raw(), CONFIG)]}


class Normalization(unittest.TestCase):
    def test_actual_committed_cp8_formats_without_relabelling(self):
        for path in (HERE.parents[1] / "benchmarks/reports").glob("*cp8*.jsonl"):
            records, errors = parse_jsonl(path.read_text(), {**CONFIG, "repetitions": 40})
            self.assertFalse(errors)
            self.assertEqual(len(records), 17)
            self.assertTrue(all(r["status"] == "passed" for r in records))
            self.assertIsNone(next(r for r in records if r["case"] == "boundary_flux_q1")["native_ratio"])

    def test_meshworker_alias_and_additional_fields(self):
        r = raw()
        r.update(case="heterogeneous_nonlinear_q1_q2_dgq0", relative_residual_error=0,
                 relative_q1_jacobian_error=0, relative_q2_jacobian_error=0,
                 relative_jacobian_action_error=0, dofs_q1=81, dofs_q2=289,
                 extra_field="preserved", quadrature_type="additional metadata")
        for s in ("median_ms", "mad_ms"):
            r["warm_dealii_meshworker_" + s] = r.pop("warm_dealii_" + s)
        result = normalize(r, CONFIG)
        self.assertEqual(result["native_ratio"], 2)
        self.assertEqual(result["raw"]["extra_field"], "preserved")

    def test_invalid_numbers_and_incomplete_fields(self):
        for key, value in [("warm_immersx_median_ms", float("nan")),
                           ("warm_dealii_median_ms", 0), ("warm_immersx_mad_ms", -1),
                           ("relative_matrix_error", None), ("relative_matrix_error", -1),
                           ("ranks", 2), ("warm_ratio", "N/A"), ("warm_ratio", None),
                           ("warm_ratio", 100), ("dofs", None), ("extra", float("inf"))]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    normalize({**raw(), key: value}, CONFIG)
        r = raw(); del r["warm_dealii_mad_ms"]
        with self.assertRaises(KeyError):
            normalize(r, CONFIG)

    def test_numerical_failure_and_extra_error_are_negative(self):
        for key in ("relative_matrix_error", "additional_error"):
            self.assertEqual(normalize({**raw(), key: 1.e-10}, CONFIG)["status"], "failed")

    def test_explicit_scope_and_operation_counts(self):
        legacy = normalize(raw(), CONFIG)
        self.assertEqual(legacy["timing_contract"]["scope"], "legacy-unspecified")
        explicit = normalize({**raw(), "timing_scope": "end_to_end",
                              "immersx_assembly_calls": 3, "native_cell_traversals": 1}, CONFIG)
        self.assertEqual(explicit["timing_contract"], {"scope": "end_to_end",
                         "immersx_assembly_calls": 3, "native_cell_traversals": 1})
        for key, value in [("timing_scope", None), ("timing_scope", ""),
                           ("immersx_assembly_calls", 1.5), ("native_cell_traversals", 0)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                normalize({**raw(), key: value}, CONFIG)

    def test_lifted_na_and_measure(self):
        r = {"case": "lifted_geometry_rebuild", "dimension": 2, "ranks": 1,
             "repetitions": 9, "measure": .2, "quadrature_points": 96,
             "warm_rebuild_median_ms": 1, "warm_rebuild_mad_ms": 0,
             "warm_bulk_assembly_median_ms": 2, "warm_bulk_assembly_mad_ms": 0,
             "native_reference": "unavailable", "immersx_over_dealii_ratio": "N/A"}
        result = normalize(r, CONFIG)
        self.assertIsNone(result["native_ratio"])
        self.assertEqual(result["mesh"]["background_dofs"], 729)
        self.assertEqual(normalize({**r, "measure": .3}, CONFIG)["status"], "failed")

    def test_bad_lines_duplicates_and_known_petsc_diagnostics(self):
        r, errors = parse_jsonl(json.dumps(raw()) + '\n{"x":NaN}\ntruncated{', CONFIG)
        self.assertEqual(len(r), 1); self.assertEqual(len(errors), 2)
        r, errors = parse_jsonl((json.dumps(raw()) + '\n') * 2, CONFIG)
        self.assertEqual(len(errors), 1)
        r, errors = parse_jsonl(json.dumps(raw()) + '\nWARNING! There are options you set that were not used!\n'
                               'Option left: name:--refinements=3 (no value) source: command line', CONFIG)
        self.assertFalse(errors)


class History(unittest.TestCase):
    def test_empty_and_missing(self):
        r = report({"source_sha": "c", "baseline": "a", "commits": ["a", "b", "c"]}, [])
        self.assertEqual(r["current"], "c")
        self.assertEqual([c["status"] for c in r["coverage"]], ["missing"] * 3)
        self.assertEqual(r["rows"], [])

    def test_multi_commit_push_recovery(self):
        commits = ["a", "b", "c", "d"]
        items = [attempt("a"), attempt("c", status="failed")]
        self.assertEqual(pending(commits, items), ["b", "d"])
        self.assertEqual(pending(commits, items, retry=True), ["b", "c", "d"])

    def test_compatible_previous_baseline_and_variability(self):
        a, b, c = attempt(), attempt("b", "2"), attempt("c", "3")
        b["records"][0]["phases"]["warm_immersx"]["median_ms"] = 3
        c["records"][0]["phases"]["warm_immersx"] = {"median_ms": 4, "mad_ms": 2}
        r = report({"source_sha": "c", "baseline": "a", "commits": ["a", "b", "c"]}, [c, a, b])
        changes = [row["changes"]["warm_immersx"] for row in r["rows"]]
        self.assertEqual(changes[1]["previous_percent"], 50)
        self.assertTrue(changes[1]["regression"])
        self.assertEqual(changes[2]["baseline_percent"], 100)
        self.assertFalse(changes[2]["regression"])

    def test_environment_and_configuration_are_partitioned(self):
        a = attempt()
        for field, key, value in [("environment", "cpu", "different"),
                                  ("environment", "image", "other"),
                                  ("configuration", "build_type", "Debug")]:
            b = copy.deepcopy(a); b[field][key] = value
            self.assertNotEqual(series_key(a, a["records"][0]), series_key(b, b["records"][0]))

    def test_timing_semantics_and_benchmark_sources_are_partitioned(self):
        a = attempt()
        for field, value in [("harness_version", 2),
                             ("benchmark_sources", {"weak_term_assembly": "new-blob"})]:
            b = {**a, field: value}
            self.assertNotEqual(series_key(a, a["records"][0]), series_key(b, b["records"][0]))
        b = copy.deepcopy(a)
        b["records"][0]["timing_contract"]["scope"] = "materialized"
        self.assertNotEqual(series_key(a, a["records"][0]), series_key(b, b["records"][0]))
        a["benchmark_sources"] = {"weak_term_assembly": "old-blob", "lifted_integration_support": "same"}
        b = copy.deepcopy(a); b["benchmark_sources"]["weak_term_assembly"] = "new-blob"
        self.assertNotEqual(series_key(a, a["records"][0]), series_key(b, b["records"][0]))
        # Library-only changes keep the benchmark blob and therefore the series.
        b = {**a, "sha": "library-change"}
        self.assertEqual(series_key(a, a["records"][0]), series_key(b, b["records"][0]))
        del b["records"][0]["timing_contract"]
        legacy = copy.deepcopy(b)
        series_key(legacy, legacy["records"][0])
        self.assertNotIn("timing_contract", legacy["records"][0])

    def test_late_attempt_cannot_hide_newer_failure(self):
        a, b = attempt(), attempt(run="2", status="failed")
        self.assertEqual(selected_attempt([b, a]), b)
        r = report({"source_sha": "a", "baseline": "a", "commits": ["a"]}, [b, a])
        self.assertEqual(r["coverage"][0]["status"], "failed")
        self.assertEqual(r["rows"][0]["changes"], {})

    def test_append_only_idempotence_retry_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = attempt()
            archive_attempt(tmp, a); archive_attempt(tmp, a)
            archive_attempt(tmp, attempt(run="2"))
            self.assertEqual(len(attempts(tmp)), 2)
            with self.assertRaises(ValueError):
                archive_attempt(tmp, {**a, "status": "failed"})
            for ident in ("../../unsafe", ""):
                with self.assertRaises(ValueError):
                    archive_attempt(tmp, {**a, "id": ident})

    def test_completed_measurement_is_not_run_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive_attempt(tmp, attempt())
            save(tmp / "plan.json", {"selected": ["a"], "configuration": CONFIG})
            with patch.dict("os.environ", {"GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1"}):
                with patch("pipeline.capture", side_effect=AssertionError("must not execute")):
                    measure(SimpleNamespace(plan=tmp / "plan.json", output=tmp))
            self.assertEqual(len(attempts(tmp)), 1)

    def test_out_of_order_plans_preserve_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            new = {"baseline": "a", "source_sha": "c", "commits": ["a", "b", "c"]}
            old = {"baseline": "a", "source_sha": "b", "commits": ["a", "b"]}
            args = SimpleNamespace(archive=tmp / "archive", incoming=tmp / "batch", plan=tmp / "plan.json")
            save(args.plan, new); merge(args)
            archive_attempt(args.incoming, attempt("b"))
            save(args.plan, old); merge(args)
            self.assertEqual(json.loads((args.archive / "plan.json").read_text()), new)
            self.assertEqual(len(attempts(args.archive)), 1)
            save(args.plan, {**new, "commits": ["a", "d"]})
            with self.assertRaises(ValueError): merge(args)

    def test_export_empty_and_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            args = SimpleNamespace(archive=tmp / "archive", plan=None, output=tmp / "site")
            export(args)
            self.assertEqual(json.loads((args.output / "history.json").read_text())["rows"], [])
            args.plan = tmp / "plan.json"
            save(args.plan, {"source_sha": "b", "baseline": "a", "commits": ["a", "b"]})
            archive_attempt(args.archive, attempt())
            export(args)
            self.assertTrue((args.output / "attempts/a-1-1.json").exists())
            self.assertEqual(json.loads((args.output / "history.json").read_text())["coverage"][1]["status"], "missing")


class GitPlanning(unittest.TestCase):
    def test_first_parent_activation_multicommit_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"; repo.mkdir()
            git(repo, "init"); git(repo, "config", "user.name", "Test")
            git(repo, "config", "user.email", "test@example.org")
            shas = []
            for i in range(4):
                if i == 1: save(repo / "scripts/performance/config.json", {})
                (repo / "test").write_text(str(i)); git(repo, "add", ".")
                git(repo, "commit", "-m", f"Synthetic commit {i}")
                shas.append(git(repo, "rev-parse", "HEAD"))
            self.assertEqual(ancestry(repo, shas[1], shas[-1]), shas[1:])
            args = SimpleNamespace(repo=repo, ref="HEAD", archive=Path(tmp) / "archive",
                output=Path(tmp) / "plan.json", preview=False, retry=False, commit=None, limit=8)
            plan(args)
            p = json.loads(args.output.read_text())
            self.assertEqual(p["baseline"], shas[1]); self.assertEqual(p["selected"], shas[1:])
            archive_attempt(args.archive, attempt(shas[2]))
            plan(args)
            self.assertEqual(json.loads(args.output.read_text())["selected"], [shas[1], shas[3]])
            args.commit = shas[0]
            with self.assertRaises(ValueError): plan(args)
            with self.assertRaises(ValueError): ancestry(repo, "not-an-ancestor", "HEAD")

    def test_activation_is_the_first_parent_merge_not_the_feature_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"; repo.mkdir()
            git(repo, "init", "-b", "master"); git(repo, "config", "user.name", "Test")
            git(repo, "config", "user.email", "test@example.org")
            (repo / "base").write_text("test"); git(repo, "add", ".")
            git(repo, "commit", "-m", "Synthetic base")
            args = SimpleNamespace(repo=repo, ref="HEAD", archive=Path(tmp) / "archive",
                output=Path(tmp) / "plan.json", preview=False, retry=False, commit=None, limit=8)
            with self.assertRaisesRegex(ValueError, "not been activated"):
                plan(args)
            git(repo, "switch", "-c", "feature")
            save(repo / "scripts/performance/config.json", {})
            git(repo, "add", "."); git(repo, "commit", "-m", "Synthetic feature activation")
            feature = git(repo, "rev-parse", "HEAD")
            git(repo, "switch", "master"); git(repo, "merge", "--no-ff", "feature", "-m", "Synthetic activation merge")
            activation = git(repo, "rev-parse", "HEAD")
            plan(args)
            p = json.loads(args.output.read_text())
            self.assertEqual(p["baseline"], activation)
            self.assertEqual(p["commits"], [activation])
            self.assertNotIn(feature, p["selected"])

    def test_read_missing_and_existing_archive_from_local_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); remote = tmp / "remote.git"; remote.mkdir()
            git(remote, "init", "--bare")
            repo = tmp / "repo"; repo.mkdir(); git(repo, "init")
            git(repo, "remote", "add", "origin", str(remote))
            archive = tmp / "first"; archive.mkdir()
            self.assertFalse(checkout(repo, archive))
            git(archive, "config", "user.name", "Test"); git(archive, "config", "user.email", "test@example.org")
            archive_attempt(archive, attempt()); git(archive, "add", ".")
            git(archive, "commit", "-m", "Synthetic archive")
            git(archive, "push", "origin", "HEAD:refs/heads/performance-data")
            other = tmp / "second"; other.mkdir()
            self.assertTrue(checkout(repo, other)); self.assertEqual(len(attempts(other)), 1)
            plan_file = tmp / "plan.json"
            save(plan_file, {"baseline": "a", "source_sha": "b", "commits": ["a", "b"]})
            batch = tmp / "batch"
            archive_attempt(batch, attempt("b", "2", "failed"))
            publish = [sys.executable, str(HERE / "archive.py"), "publish", "--repo", str(repo),
                       "--incoming", str(batch), "--plan", str(plan_file)]
            local_env = {**os.environ, "GITHUB_ACTIONS": "false"}
            subprocess.run(publish, check=True, capture_output=True, env=local_env)
            first_head = git(remote, "rev-parse", "refs/heads/performance-data")
            subprocess.run(publish, check=True, capture_output=True, env=local_env)
            self.assertEqual(git(remote, "rev-parse", "refs/heads/performance-data"), first_head)
            final = tmp / "final"; final.mkdir(); checkout(repo, final)
            self.assertEqual(len(attempts(final)), 2)


class Publication(unittest.TestCase):
    def test_only_trusted_master_batches_can_be_written_in_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            args = SimpleNamespace(plan=tmp / "plan.json", incoming=tmp / "batch")
            p = {"source_sha": "master-head", "harness_sha": "master-head", "selected": ["a"]}
            save(args.plan, p)
            a = {**attempt(), "harness_sha": "master-head"}
            archive_attempt(args.incoming, a)
            env = {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/master",
                   "GITHUB_EVENT_NAME": "push", "GITHUB_SHA": "master-head",
                   "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1"}
            for event in ("push", "schedule", "workflow_dispatch"):
                with patch.dict(os.environ, {**env, "GITHUB_EVENT_NAME": event}):
                    validate_publication(args)
            for key, value in [("GITHUB_REF", "refs/heads/feature"),
                               ("GITHUB_EVENT_NAME", "pull_request"),
                               ("GITHUB_EVENT_NAME", "pull_request_target"),
                               ("GITHUB_SHA", "different"), ("GITHUB_RUN_ID", "2"),
                               ("GITHUB_RUN_ATTEMPT", "2")]:
                with self.subTest(key=key, value=value), patch.dict(os.environ, {**env, key: value}):
                    with self.assertRaises(ValueError):
                        validate_publication(args)
            for field, value in [("source_sha", "different"), ("harness_sha", "different"),
                                 ("selected", ["not-a"])]:
                save(args.plan, {**p, field: value})
                with patch.dict(os.environ, env), self.assertRaises(ValueError):
                    validate_publication(args)
            # No synthetic test archive or local measurement is sent to origin.
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}):
                validate_publication(args)

    def test_raw_conflicts_are_rejected_without_changing_archived_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            args = SimpleNamespace(plan=tmp / "plan.json", incoming=tmp / "batch", archive=tmp / "archive")
            save(args.plan, {"baseline": "a", "source_sha": "a", "commits": ["a"]})
            source = args.incoming / "raw/a-1-1/output.jsonl"
            source.parent.mkdir(parents=True); source.write_text("original")
            merge(args); merge(args)
            source.write_text("conflict")
            with self.assertRaisesRegex(ValueError, "conflicting raw output"):
                merge(args)
            self.assertEqual((args.archive / "raw/a-1-1/output.jsonl").read_text(), "original")


if __name__ == "__main__":
    unittest.main()
