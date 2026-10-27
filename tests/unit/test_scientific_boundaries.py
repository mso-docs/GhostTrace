"""Hand-computable scientific edge cases; no model, network, or invented results."""
import copy
import math
import unittest
from ghosttrace.analysis.metrics import distances, lexical_counts
from ghosttrace.analysis.bootstrap import bootstrap_scenarios
from ghosttrace.config import EngineError, validate_config, validate_dataset, canonical
from ghosttrace.engine.render import align_continuation
from test_engine import CONFIG, Tokenizer


class MeasurementBoundaries(unittest.TestCase):
    def test_vector_symmetry_scaling_and_opposition(self):
        a, b = [3., 4.], [0., 5.]
        forward, reverse = distances(a, b), distances(b, a)
        self.assertAlmostEqual(forward['normalized_l2'], reverse['normalized_l2'])
        self.assertAlmostEqual(forward['cosine_distance'], reverse['cosine_distance'])
        self.assertAlmostEqual(distances([30., 40.], [0., 50.])['normalized_l2'], forward['normalized_l2'], places=7)
        opposed = distances([1., 0.], [-1., 0.])
        self.assertAlmostEqual(opposed['cosine_distance'], 2.)
        self.assertAlmostEqual(opposed['normalized_l2'], 2., places=7)

    def test_zero_is_a_valid_identity_and_cosine_is_missing(self):
        zero = distances([0., 0.], [0., 0.])
        self.assertEqual(zero['normalized_l2'], 0.)
        self.assertIsNone(zero['cosine_distance'])
        self.assertEqual(zero['cosine_missing_reason'], 'zero_norm')

    def test_malformed_vectors_never_become_metrics(self):
        for a, b in [([], []), ([1], [1, 2]), ([math.nan], [1]), ([1], [math.inf]), ([-math.inf], [0])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                distances(a, b)

    def test_lexical_proxy_word_boundaries_and_interpretation(self):
        values = lexical_counts('Sorry! I apologize; not my fault. Replacement is not replace.')
        self.assertEqual(values['counts']['regret'], 2)
        self.assertEqual(values['counts']['deflection'], 1)
        self.assertEqual(values['counts']['repair'], 1)
        self.assertIn('not a remorse score', values['interpretation'])

    def test_bootstrap_null_singleton_and_all_zero(self):
        empty = bootstrap_scenarios([])
        self.assertEqual(empty['n'], 0)
        self.assertIsNone(empty['mean'])
        one = bootstrap_scenarios([.25])
        self.assertEqual((one['mean'], one['lower'], one['upper'], one['n']), (.25, .25, .25, 1))
        zeros = bootstrap_scenarios([0.] * 4)
        self.assertEqual((zeros['mean'], zeros['lower'], zeros['upper']), (0., 0., 0.))
        self.assertEqual(zeros['unit'], 'scenario')

    def test_boundary_tokenization_is_rejected(self):
        class BoundaryTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2] if text == 'prefix' else [1, 99] + list(range(16))
        with self.assertRaises(EngineError) as error:
            align_continuation(BoundaryTokenizer(), 'prefix', 'continuation')
        self.assertEqual(error.exception.code, 'TOKEN_ALIGNMENT_FAILED')

    def test_short_continuation_cannot_silently_shrink_mask(self):
        with self.assertRaises(EngineError) as error:
            align_continuation(Tokenizer(), 'prefix', 'fifteen tokens!')
        self.assertEqual(error.exception.code, 'TOKEN_ALIGNMENT_FAILED')

    def test_dataset_limits_and_duplicate_ids(self):
        row = {'id':'a', 'family_id':'f', 'split':'development', 'text':'t', 'common_continuation':'c'}
        for rows in [[], [row, row], [{**row, 'text':'x' * 2001}], [{**row, 'split':'training'}], [{**row, 'extra':'injected'}]]:
            with self.subTest(rows=rows[:1]), self.assertRaises(EngineError):
                validate_dataset(rows)

    def test_config_limits_and_unknown_nested_settings(self):
        cases = [('generation', 'max_new_tokens', 97), ('limits', 'batch_size', 2), ('limits', 'max_artifact_bytes', 0), ('capture', 'storage_dtype', 'float16'), ('analysis', 'bootstrap_draws', 100), ('generation', 'temperature', .7)]
        for section, key, value in cases:
            config = copy.deepcopy(CONFIG)
            config[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(EngineError):
                validate_config(config)

    def test_canonical_json_retains_unicode_and_rejects_nonfinite_nested(self):
        self.assertIn('é'.encode(), canonical({'text':'é'}))
        with self.assertRaises(ValueError):
            canonical({'nested':[{'bad':float('-inf')}]})
