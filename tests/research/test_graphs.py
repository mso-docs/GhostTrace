import unittest
from ghosttrace.research.service import merge_graph

class GraphIntegrity(unittest.TestCase):
    def test_external_node_edge_collision_cannot_override_native_observations(self):
        capture={'id':'capture','condition_id':'clean','status':'completed','components':[{'site':'resid_post','layer':0,'array_key':'L0.resid_post'}],'graph':{'nodes':[{'id':'L0.resid_post','site':'resid_post','layer':0,'label':'Native residual','kind':'component','measured':True},{'id':'logits','kind':'logits'}],'edges':[{'id':'native-edge','source':'L0.resid_post','target':'logits','kind':'structural'}]}}
        external={'id':'external-record','source':'external_import','graph':{'nodes':[{'id':'L0.resid_post','kind':'feature','label':'Invented concept','site':'invented','measured':True},{'id':'logits','kind':'target'}],'edges':[{'id':'native-edge','source':'L0.resid_post','target':'logits','kind':'attribution'}]}}
        graph=merge_graph([capture],[],circuits=[external]);native=next(n for n in graph['nodes'] if n['id']=='L0.resid_post')
        self.assertEqual(native['label'],'Native residual');self.assertEqual(native['site'],'resid_post');self.assertEqual(len(native['observation_refs']),1)
        imported=next(n for n in graph['nodes'] if n['id'].startswith('external:external-record:L0'))
        self.assertFalse(imported['measured']);self.assertEqual(imported['observation_refs'],[])
        self.assertEqual(len(graph['edges']),2);self.assertTrue(any(e['evidence_origin']=='external_reported' for e in graph['edges']))
    def test_multiple_conditions_and_later_intervention_preserve_all_observation_refs(self):
        node={'id':'L0.resid_post','site':'resid_post','layer':0,'kind':'component','measured':True}
        captures=[{'id':name,'condition_id':name,'status':'completed','components':[{'site':'resid_post','layer':0,'array_key':'L0.resid_post'}],'graph':{'nodes':[node],'edges':[]}} for name in ['a','b']]
        intervention={'id':'effect','status':'completed','spec':{'kind':'exact','site':'resid_post','layer':0,'operation':'patch'},'evidence_type':'intervention','raw_delta':-.5,'graph':{'nodes':[{**node,'measured':False,'intervention_effect':-.5}],'edges':[]}}
        merged=merge_graph(captures,[intervention])['nodes'][0]
        self.assertTrue(merged['measured']);self.assertEqual({o['condition_id'] for o in merged['observation_refs']},{'a','b'});self.assertEqual(merged['intervention_refs'][0]['delta'],-.5);self.assertNotIn('intervention_effect',merged)
