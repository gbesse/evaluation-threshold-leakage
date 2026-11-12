import unittest,copy,json
from evaluation_threshold_leakage import run
DEMO={'thresholds': [0.5, 0.7], 'validation': [{'id': 'v1', 'score': 0.8, 'label': True}, {'id': 'v2', 'score': 0.6, 'label': False}], 'test': [{'id': 't1', 'score': 0.6, 'label': True}, {'id': 't2', 'score': 0.4, 'label': False}]}
class DomainTests(unittest.TestCase):
 def test_oracle_is_optimistic_on_same_test(self):
  x=run(DEMO);self.assertEqual(x['validation_selected_threshold'],.7);self.assertEqual(x['heldout_accuracy'],.5);self.assertEqual(x['test_selected_oracle_threshold'],.5);self.assertEqual(x['observed_selection_advantage'],.5)
 def test_test_not_used_for_actual_selection(self):
  self.assertFalse(run(DEMO)['test_used_for_actual_selection']);self.assertFalse(run(DEMO)['generalization_bias_estimated'])
 def test_overlap_rejected(self):
  x=copy.deepcopy(DEMO);x['test'][0]['id']='v1'
  with self.assertRaises(ValueError):run(x)
