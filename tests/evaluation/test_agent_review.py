"""Evidence-grounded agent-review contracts; a review cannot self-award release."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('agent_review',Path(__file__).parents[2]/'scripts/agent_review.py')
reviewer = importlib.util.module_from_spec(spec);spec.loader.exec_module(reviewer)


class AgentReviewContracts(unittest.TestCase):
    def review(self,root):
        evidence = root/'evidence.txt';evidence.write_text('observed and tested')
        packet = root/'packet.json';packet.write_text(json.dumps({'evidence_inventory':[{'path':'evidence.txt','sha256':reviewer.sha(evidence)}]}))
        record = {'schema_version':'1.0','packet_sha256':reviewer.sha(packet),'reviewer':{'name':'independent tester','kind':'agent','model':'test-model','reviewed_at':'2026-10-27T00:00:00Z'},'criteria':{k:{'status':'pass','reason':'Inspected and reproduced evidence.','evidence':[{'path':'evidence.txt','sha256':reviewer.sha(evidence),'locator':'line 1'}]} for k in reviewer.CRITERIA},'findings':[],'recommendation':'ship'}
        return packet,record

    def test_valid_review_and_checksum_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);packet,record=self.review(root);path=root/'review.json';path.write_text(json.dumps(record))
            self.assertEqual(reviewer.validate(path,packet,root=root)['status'],'pass')
            (root/'evidence.txt').write_text('changed after review')
            self.assertEqual(reviewer.validate(path,packet,root=root)['status'],'fail')

    def test_missing_evidence_or_conflicting_ship_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);packet,record=self.review(root);path=root/'review.json'
            record['criteria']['scientific_honesty']['evidence']=[];path.write_text(json.dumps(record))
            self.assertEqual(reviewer.validate(path,packet,root=root)['status'],'fail')
            record['criteria']['scientific_honesty']['status']='blocked';path.write_text(json.dumps(record))
            self.assertEqual(reviewer.validate(path,packet,root=root)['status'],'fail')
            record['criteria']['scientific_honesty']['status']='pass';record['criteria']['scientific_honesty']['evidence']=[{'path':'evidence.txt','sha256':reviewer.sha(root/'evidence.txt'),'locator':'line 1'}]
            record['criteria']['ratings_and_holdout']['status']='blocked'
            record['recommendation']='human_review';path.write_text(json.dumps(record))
            result=reviewer.validate(path,packet,root=root)
            self.assertEqual(result['status'],'pass');self.assertTrue(result['review_completed']);self.assertFalse(result['release_approved']);self.assertTrue(result['requires_human_review'])

    def test_paths_outside_checkout_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);packet,record=self.review(root);path=root/'review.json'
            record['criteria']['scientific_honesty']['evidence'][0]['path']='../outside'
            path.write_text(json.dumps(record))
            self.assertEqual(reviewer.validate(path,packet,root=root)['status'],'fail')
