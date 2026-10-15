"""Version 1 performance records. No third-party Python dependencies."""
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = 1
PENALTY_CASES = {f"particle_penalty_{sd}d_in_{d}d_{kind}": d
                 for sd, d in ((1, 2), (2, 3), (1, 3), (2, 2), (3, 3))
                 for kind in ("scalar", "vector")}
EQUIVALENT = {"rhs_constant", "rhs_boundary", "mass_q1", "laplace_q1",
              "nonmatching_value_forward", "nonmatching_value_reverse",
              "heterogeneous_nonlinear_q1_q2_dgq0"}
ERRORS = {
    "rhs_constant": ["relative_rhs_error"],
    "rhs_boundary": ["relative_rhs_error"],
    "boundary_flux_q1": ["relative_rhs_error"],
    "heterogeneous_nonlinear_q1_q2_dgq0": ["relative_residual_error",
        "relative_q1_jacobian_error", "relative_q2_jacobian_error",
        "relative_jacobian_action_error"],
    "independent_background_trace_mass": ["analytical_measure_error", "analytical_x2_error"],
    "independent_rotated_trace_surface_3d": ["analytical_area_error"],
}
EXPECTED = [(c, d) for d in (1, 2, 3)
            for c in ("rhs_constant", "mass_q1", "laplace_q1")]
EXPECTED += [(c, d) for d in (2, 3) for c in ("rhs_boundary", "boundary_flux_q1")]
EXPECTED += [(c, 2) for c in ("heterogeneous_nonlinear_q1_q2_dgq0",
             "nonmatching_value_forward", "nonmatching_value_reverse",
             "independent_background_trace_mass", "lifted_geometry_rebuild")]
EXPECTED += [("independent_rotated_trace_surface_3d", 3)]
EXPECTED += list(PENALTY_CASES.items())
EQUIVALENT.update(PENALTY_CASES)
for case in PENALTY_CASES:
    ERRORS[case] = ["relative_matrix_error", "max_entry_error", "relative_action_error",
                    "relative_native_manual_error", "native_max_entry_error", "constant_measure_error"]


def expected_cases(configuration):
    return [(case, dim, rank) for case, dim in EXPECTED
            for rank in (configuration.get("particle_penalty_ranks", [1])
                         if case in PENALTY_CASES else [1])]


def dumps(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: fail("nonfinite " + x))


def fail(message):
    raise ValueError(message)


def number(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        fail("expected a number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        fail("invalid numeric value")
    return value


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        fail("nonfinite value")
    if isinstance(value, dict):
        for v in value.values():
            finite_tree(v)
    if isinstance(value, list):
        for v in value:
            finite_tree(v)


def normalize(raw, configuration):
    """Adapt CP8 JSONL; unknown fields survive in raw, not as implicit success."""
    finite_tree(raw)
    case = raw["case"]
    if (case, raw["dimension"]) not in EXPECTED:
        fail("unsupported case/dimension")
    for key in ("dimension", "ranks", "repetitions"):
        v = number(raw[key], True)
        if int(v) != v:
            fail("noninteger " + key)
    allowed_ranks = configuration.get("particle_penalty_ranks", [1]) if case in PENALTY_CASES else [1]
    if raw["ranks"] not in allowed_ranks or raw["repetitions"] != configuration["repetitions"]:
        fail("unexpected rank/repetition count")
    checks = {}
    if case in PENALTY_CASES:
        support_dim = int(case.split("_")[2][:-1])
        components = raw["dimension"] if case.endswith("_vector") else 1
        if raw["support_dimension"] != support_dim or raw["components"] != components:
            fail("penalty case metadata mismatch")
        number(raw["penalty"], True)
        owners = number(raw["active_particle_ranks"], True)
        if int(owners) != owners or not min(2, raw["ranks"]) <= owners <= raw["ranks"]:
            fail("penalty fixture does not exercise distributed particle ownership")
    if case == "lifted_geometry_rebuild":
        checks["measure_error"] = {"value": abs(number(raw["measure"], True) - 0.2),
                                   "tolerance": 1.e-10}
    else:
        for key in ERRORS.get(case, ["relative_matrix_error"]):
            checks[key] = {"value": number(raw[key]), "tolerance": 1.e-12}
    # Extra correctness fields from newer benchmarks must not hide failures.
    for key, value in raw.items():
        if key.endswith("_error") and key not in checks:
            checks[key] = {"value": number(value), "tolerance": 1.e-12}
    phases = {}
    for key, value in raw.items():
        if key.endswith("_median_ms"):
            prefix = key[:-10]
            phases[prefix] = {"median_ms": number(value, True),
                              "mad_ms": number(raw[prefix + "_mad_ms"])}
        elif key.endswith("_ms"):
            number(value)
    if case in PENALTY_CASES:
        required_phases = {"warm_immersx", "warm_dealii"}
        required_mesh = {"dofs", "cells", "support_cells", "support_dimension",
                         "quadrature_points_per_cell", "components"}
    elif case == "lifted_geometry_rebuild":
        required_phases = {"warm_rebuild", "warm_bulk_assembly"}
        required_mesh = {"quadrature_points"}
    elif case.startswith("independent_"):
        required_phases = {"warm_immersx_assembly", "operator_action"}
        required_mesh = {"background_dofs", "background_cells", "support_cells"}
    elif case == "heterogeneous_nonlinear_q1_q2_dgq0":
        required_phases = {"warm_immersx", "warm_dealii_meshworker"}
        required_mesh = {"dofs_q1", "dofs_q2", "cells", "quadrature_points_per_cell"}
    elif case.startswith("nonmatching_"):
        required_phases = {"warm_immersx", "warm_dealii"}
        required_mesh = {"bulk_dofs", "embedded_dofs"}
    elif case in {"rhs_constant", "rhs_boundary", "boundary_flux_q1", "mass_q1", "laplace_q1"}:
        required_phases = {"warm_immersx", "warm_dealii"}
        required_mesh = {"dofs", "cells"}
    if not required_phases <= phases.keys() or not required_mesh <= raw.keys():
        fail("missing timing phase or mesh metadata")
    native = phases.get("warm_dealii", phases.get("warm_dealii_meshworker"))
    equivalent = case in EQUIVALENT and raw.get("timing_equivalent", True) is True
    ratio = None
    if equivalent:
        if native is None or "warm_immersx" not in phases:
            fail("missing equivalent native timing")
        ratio = phases["warm_immersx"]["median_ms"] / native["median_ms"]
        for key in ("warm_ratio", "native_timing_ratio", "immersx_over_dealii_ratio"):
            if key in raw:
                # A claimed equivalent comparison must contain a numeric ratio.
                reported = number(raw[key], True)
                if not math.isclose(reported, ratio, rel_tol=5.e-5):
                    fail("inconsistent native ratio")
    reason = None if equivalent else raw.get("timing_note", "No equivalent native bulk–bulk assembler")
    mesh_keys = {"dofs", "cells", "bulk_dofs", "embedded_dofs", "background_dofs",
                 "background_cells", "support_cells", "dofs_q1", "dofs_q2",
                 "quadrature_points_per_cell", "quadrature_points", "dimension",
                 "embedded_dimension", "support_dimension", "representative_dimension",
                 "spacedimension", "refinements", "components"}
    mesh = {k: v for k, v in raw.items() if k in mesh_keys}
    for key, value in mesh.items():
        number(value, key != "refinements")
        if int(value) != value:
            fail("noninteger mesh metadata")
    if not any("dofs" in k for k in mesh):
        if case != "lifted_geometry_rebuild":
            fail("missing DoF count")
        # CP8 omits the Q1 background counts; these follow its uniform cube.
        mesh["background_dofs"] = (2 ** configuration["refinements"] + 1) ** 3
        mesh["background_cells"] = 8 ** configuration["refinements"]
    mesh["quadrature_recipe"] = configuration["quadrature_recipe"]
    timing_contract = {"scope": raw.get("timing_scope", "legacy-unspecified")}
    if case in PENALTY_CASES:
        timing_contract["penalty"] = raw["penalty"]
    if not isinstance(timing_contract["scope"], str) or not timing_contract["scope"]:
        fail("invalid timing scope")
    for key in ("immersx_assembly_calls", "native_cell_traversals"):
        if key in raw:
            value = number(raw[key], True)
            if int(value) != value:
                fail("noninteger operation count")
            timing_contract[key] = value
    return {"timing_contract": timing_contract,
            "case": case, "dimension": raw["dimension"], "ranks": raw["ranks"],
            "mesh": mesh, "phases": phases, "checks": checks,
            "status": "passed" if all(c["value"] < c["tolerance"] for c in checks.values()) else "failed",
            "native_ratio": ratio, "native_na_reason": reason, "raw": raw}


def parse_jsonl(text, configuration):
    records, errors = [], []
    seen = set()
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        # PETSc reports CLI options consumed by the benchmark rather than PETSc.
        # Ignore only these exact diagnostics; preserve them in the raw stream.
        if line in ("WARNING! There are options you set that were not used!",
                    "WARNING! could be spelling mistake, etc!") or re.fullmatch(
                r"There are [12] unused database options\. They are:|"
                r"Option left: name:--(?:refinements|repetitions)=\d+ \(no value\) source: command line", line):
            continue
        try:
            raw = json.loads(line, parse_constant=lambda x: fail("nonfinite " + x))
            record = normalize(raw, configuration)
            key = (record["case"], record["dimension"], record["ranks"])
            if key in seen:
                fail("duplicate case")
            seen.add(key)
            records.append(record)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"line": line_no, "reason": str(exc)})
    return records, errors


def series_key(attempt, record):
    value = {"environment": attempt["environment"], "configuration": attempt["configuration"],
             "harness_version": attempt["harness_version"], "case": record["case"],
             "mesh": record["mesh"], "ranks": record["ranks"]}
    # Legacy archives are not rewritten or assigned inferred timing semantics.
    if "timing_contract" in record:
        value["timing_contract"] = record["timing_contract"]
    if "benchmark_sources" in attempt:
        if record["case"] in PENALTY_CASES:
            name = "particle_penalty"
            value["native_oracle_sources"] = {key: attempt["benchmark_sources"][key]
                for key in ("particle_penalty_native", "particle_penalty_coupling")}
        elif record["case"] == "lifted_geometry_rebuild":
            name = "lifted_integration_support"
        else:
            name = "weak_term_assembly"
        value["benchmark_source"] = attempt["benchmark_sources"][name]
    return hashlib.sha256(dumps(value).encode()).hexdigest()[:20]


def archive_attempt(directory, attempt):
    """Append-only identity; a retry has another attempt, never overwrites."""
    path = Path(directory) / "attempts" / (attempt["id"] + ".json")
    if not attempt["id"] or not all(c.isalnum() or c in "-_" for c in attempt["id"]):
        fail("unsafe attempt id")
    payload = dumps(attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_text() != payload:
            fail("conflicting immutable attempt " + attempt["id"])


def attempts(directory):
    return [load(p) for p in sorted((Path(directory) / "attempts").glob("*.json"))]


def selected_attempt(items):
    # Most recent *started* attempt wins, including failures. Late completion
    # cannot replace a newer attempt or conceal a newer validation failure.
    return max(items, key=lambda x: (x["started_at"], x["run"], x["attempt"])) if items else None


def pending(commits, items, retry=False):
    result = []
    for sha in commits:
        selected = selected_attempt([a for a in items if a["sha"] == sha])
        if selected is None or (retry and selected["status"] != "passed"):
            result.append(sha)
    return result


def report(plan, items):
    previous, baseline, rows, coverage = {}, {}, [], []
    for sha in plan["commits"]:
        found = [a for a in items if a["sha"] == sha]
        selected = selected_attempt(found)
        coverage.append({"sha": sha, "status": selected["status"] if selected else "missing",
                         "attempts": [a["id"] for a in found]})
        if not selected:
            continue
        for r in selected["records"]:
            key = series_key(selected, r)
            row = {**r, "sha": sha, "id": selected["id"], "series": key,
                   "environment": selected["environment"], "configuration": selected["configuration"],
                   "run_url": selected["run_url"], "started_at": selected["started_at"],
                   "attempt_status": selected["status"],
                   "changes": {}}
            for phase, timing in r["phases"].items():
                pk = (key, phase)
                if selected["status"] != "passed" or r["status"] != "passed":
                    continue
                old, first = previous.get(pk), baseline.get(pk)
                delta = timing["median_ms"] - old["median_ms"] if old else None
                row["changes"][phase] = {
                    "previous_percent": 100 * delta / old["median_ms"] if old else None,
                    "baseline_percent": 100 * (timing["median_ms"] / first["median_ms"] - 1) if first else None,
                    "previous_sha": old["sha"] if old else None,
                    "baseline_sha": first["sha"] if first else None,
                    "regression": bool(old and delta > max(0.1, 0.1 * old["median_ms"],
                                                           3 * (old["mad_ms"] + timing["mad_ms"])))}
                previous[pk] = {**timing, "sha": sha}
                baseline.setdefault(pk, {**timing, "sha": sha})
            rows.append(row)
    return {"schema_version": SCHEMA, "source_sha": plan["source_sha"],
            "baseline": plan["baseline"], "current": plan["commits"][-1] if plan["commits"] else None,
            "coverage": coverage, "rows": rows, "attempts": items}
