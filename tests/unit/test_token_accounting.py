"""Token accounting is metadata-only and must not change input IDs."""
import unittest
from ghosttrace.engine.render import token_accounting

class ChatTokenizer:
    all_special_ids=[ord('|')]
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
    def __call__(self,text,**kwargs):return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        text=''.join('|'+m['role']+':'+m['content']+'|' for m in messages)
        return text+('|assistant:' if add_generation_prompt else '')

class TokenAccounting(unittest.TestCase):
    def test_complete_roles_special_masks_and_fixed_continuation(self):
        tokenizer=ChatTokenizer();messages=[{'role':'system','content':'Instruction'},{'role':'user','content':'Scenario'}]
        rendered=tokenizer.apply_chat_template(messages,add_generation_prompt=True)
        result=token_accounting(tokenizer,rendered,messages,' continuation')
        self.assertEqual(result['role_span_mapping'],'complete_chat_template_envelopes')
        self.assertEqual([r['span_kind'] for r in result['role_spans']],['complete_message_envelope','complete_message_envelope','generation_prompt','fixed_continuation'])
        self.assertEqual([r['role'] for r in result['role_spans']],['system','user','assistant','assistant'])
        self.assertEqual(len(result['special_token_mask']),len(tokenizer.encode(rendered+' continuation')))
        self.assertEqual(sum(result['special_token_mask']),5)
        self.assertEqual(result['role_spans'][0]['character_start'],0)
        covered={i for span in result['role_spans'] for i in span['token_indices']}
        self.assertEqual(covered,set(range(len(tokenizer.encode(rendered+' continuation')))))
