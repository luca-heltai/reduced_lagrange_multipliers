"""Synthetic schema fixtures only; never imported as performance measurements."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history import PENALTY_CASES, expected_cases, normalize, parse_jsonl, report, series_key
from test_history import CONFIG, attempt, raw


def penalty_raw(case):
    support_dimension = int(case.split('_')[2][0])
    dimension = PENALTY_CASES[case]
    return {**raw(), "case": case, "dimension": dimension,
            "support_dimension": support_dimension, "support_cells": 2 ** support_dimension,
            "components": dimension if case.endswith('vector') else 1,
            "quadrature_points_per_cell": 3 ** support_dimension,
            "penalty": 2.75, "relative_native_manual_error": 0, "native_max_entry_error": 0,
            "max_entry_error": 0, "relative_action_error": 0,
            "constant_measure_error": 0, "active_particle_ranks": 1,
            "timing_equivalent": True,
            "timing_scope": "particle_generation_search_and_materialized_matrix"}


class ParticlePenalty(unittest.TestCase):
    def test_all_scalar_vector_and_codimension_cases_have_native_ratio(self):
        for case in PENALTY_CASES:
            record = normalize(penalty_raw(case), CONFIG)
            self.assertEqual(record['native_ratio'], 2)
            self.assertEqual(record['status'], 'passed')
            self.assertEqual(len(record['checks']), 6)

    def test_every_required_correctness_check_is_negative_on_failure(self):
        case = next(iter(PENALTY_CASES))
        raw = penalty_raw(case)
        for key in ('relative_native_manual_error', 'native_max_entry_error',
                    'relative_matrix_error', 'max_entry_error', 'relative_action_error', 'constant_measure_error'):
            self.assertEqual(normalize({**raw, key: 1.e-9}, CONFIG)['status'], 'failed')
            missing = dict(raw); del missing[key]
            with self.assertRaises(KeyError):
                normalize(missing, CONFIG)

    def test_mismatched_case_metadata_and_invalid_penalties_are_rejected(self):
        raw = penalty_raw("particle_penalty_1d_in_3d_vector")
        for key, value in (("components", 1), ("support_dimension", 2),
                           ("penalty", 0), ("penalty", None)):
            with self.assertRaises(ValueError):
                normalize({**raw, key: value}, CONFIG)

    def test_mpi_ownership_and_rank_counts_are_required(self):
        case = next(iter(PENALTY_CASES))
        cfg = {**CONFIG, 'particle_penalty_ranks': [1, 2]}
        mpi = {**penalty_raw(case), 'ranks': 2, 'active_particle_ranks': 2}
        self.assertEqual(normalize(mpi, cfg)['status'], 'passed')
        for update in ({'active_particle_ranks': 1}, {'active_particle_ranks': 3},
                       {'active_particle_ranks': 1.5}, {'ranks': 4}):
            with self.assertRaises(ValueError):
                normalize({**mpi, **update}, cfg)
        serial = normalize(penalty_raw(case), cfg)
        self.assertNotEqual(series_key(attempt(), serial), series_key(attempt(), normalize(mpi, cfg)))

    def test_serial_and_mpi_expected_coverage_and_duplicate_detection(self):
        cfg = {**CONFIG, 'particle_penalty_ranks': [1, 2]}
        self.assertEqual(len(expected_cases(cfg)), 39)
        lines = [json.dumps({**penalty_raw(case), 'ranks': rank, 'active_particle_ranks': rank})
                 for case in PENALTY_CASES for rank in (1, 2)]
        records, errors = parse_jsonl('\n'.join(lines), cfg)
        self.assertFalse(errors); self.assertEqual(len(records), 20)
        _, errors = parse_jsonl('\n'.join(lines + [lines[-1]]), cfg)
        self.assertEqual(len(errors), 1)

    def test_mesh_metadata_and_all_native_sources_partition_series(self):
        case = next(iter(PENALTY_CASES))
        a = attempt()
        a['records'] = [normalize(penalty_raw(case), CONFIG)]
        a['benchmark_sources'] = {'particle_penalty': 'synthetic-cc',
            'particle_penalty_native': 'synthetic-manual', 'particle_penalty_coupling': 'synthetic-coupling'}
        for key in a['benchmark_sources']:
            b = copy.deepcopy(a); b['benchmark_sources'][key] = 'different-synthetic-source'
            self.assertNotEqual(series_key(a, a['records'][0]), series_key(b, b['records'][0]))
        b = copy.deepcopy(a); b['records'][0]['mesh']['components'] = 3
        self.assertNotEqual(series_key(a, a['records'][0]), series_key(b, b['records'][0]))

    def test_jsonl_export_contains_chart_rows_and_equivalent_ratio(self):
        a = attempt()
        records, errors = parse_jsonl('\n'.join(json.dumps(penalty_raw(case)) for case in PENALTY_CASES), CONFIG)
        self.assertFalse(errors); self.assertEqual(len(records), 10)
        a['records'] = records
        data = report({'source_sha': 'a', 'baseline': 'a', 'commits': ['a']}, [a])
        self.assertEqual(len(data['rows']), 10)
        self.assertTrue(all(row['native_ratio'] == 2 for row in data['rows']))
        self.assertTrue(all('warm_immersx' in row['changes'] for row in data['rows']))


if __name__ == '__main__':
    unittest.main()
