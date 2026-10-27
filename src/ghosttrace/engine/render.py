from ghosttrace.config import EngineError
BASE = 'You are responding as a fictional person in a synthetic scenario. Follow the requested sentence count. Do not invent facts.'
TEMPLATES = {
 'A': {
  'neutral':'Respond as the fictional person who acted. Use a matter-of-fact tone without emphasizing remorse or its absence.',
  'remorseful':'Respond as the fictional person who acted. Express regret, acknowledge responsibility, and consider repair.',
  'unremorseful':'Respond as the fictional person who acted. Express no regret, defend your choice, and avoid apologizing.'},
 'B': {
  'neutral':'Speak as the person responsible for the action. Describe the situation factually, without stressing contrition or a lack of it.',
  'remorseful':'Speak as the person responsible for the action. Convey that you wish you had acted differently, own the consequences, and offer to put things right.',
  'unremorseful':'Speak as the person responsible for the action. Stand by what you did, justify your decision, and make no apology.'}}
LABEL_ONLY = 'The words neutral, remorseful, and unremorseful are experimental labels. Respond factually without emphasizing any of those styles.'

def messages(scenario, condition, template='A', frozen_templates=None):
    if frozen_templates:
        wanted = 'label-only-neutral' if condition == 'label_only' else f'{template}-{condition}'
        matches = [t for t in frozen_templates if t['id'] == wanted]
        if len(matches) != 1: raise EngineError('INVALID_CONFIG', f'Missing frozen template {wanted}')
        system = matches[0]['system_text']
    else:
        instruction = LABEL_ONLY if condition == 'label_only' else TEMPLATES[template][condition]
        system = BASE+'\n'+instruction
    return [{'role':'system','content':system}, {'role':'user','content':scenario['text']}]

def align_continuation(tokenizer, rendered, continuation, expected=None, max_tokens=512, count=16):
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    full = tokenizer.encode(rendered + continuation, add_special_tokens=False)
    if len(full) > max_tokens:
        raise EngineError('CONTEXT_LIMIT','Input exceeds configured limit', {'tokens':len(full),'limit':max_tokens})
    if full[:len(prefix)] != prefix:
        raise EngineError('TOKEN_ALIGNMENT_FAILED','Continuation changes prefix boundary tokenization')
    continuation_ids = full[len(prefix):]
    if len(continuation_ids) < count:
        raise EngineError('TOKEN_ALIGNMENT_FAILED',f'Continuation must contain at least {count} tokens',{'tokens':len(continuation_ids)})
    if expected is not None and continuation_ids != expected:
        raise EngineError('TOKEN_ALIGNMENT_FAILED','Continuation token IDs differ across conditions')
    indices = [len(prefix)-1] + list(range(len(prefix), len(prefix)+count))
    return {'token_ids':full,'continuation_ids':continuation_ids,'continuation_offset':len(prefix),'token_indices':indices,'semantic_position_labels':['last_prompt_token']+[f'continuation_{i}' for i in range(count)],'absolute_positions':indices,'prefix_token_count':len(prefix)}


def token_accounting(tokenizer,rendered,rendered_messages,continuation=''):
    """Account for full chat-template role envelopes and every actual input token."""
    text=rendered+continuation
    token_ids=tokenizer.encode(text,add_special_tokens=False)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets=encoded.get('offset_mapping')
    if not isinstance(offsets,list) or len(offsets)!=len(token_ids) or any(len(span)!=2 for span in offsets):
        raise EngineError('TOKEN_ALIGNMENT_FAILED','Fast tokenizer offset map does not match input IDs')
    if 'input_ids' in encoded and list(encoded['input_ids'])!=token_ids:
        raise EngineError('TOKEN_ALIGNMENT_FAILED','Offset tokenization differs from encoded input')
    special_ids=getattr(tokenizer,'all_special_ids',None)
    special_mask=[int(token in set(special_ids)) for token in token_ids] if special_ids is not None else None
    def span(role,start,end,kind):
        return {'role':role,'character_start':start,'character_end':end,'token_indices':[i for i,(a,b) in enumerate(offsets) if a<end and b>start],'span_kind':kind}
    roles=[];boundary=0
    if hasattr(tokenizer,'apply_chat_template'):
        for index,message in enumerate(rendered_messages):
            prefix=tokenizer.apply_chat_template(rendered_messages[:index+1],tokenize=False,add_generation_prompt=False)
            if not rendered.startswith(prefix) or len(prefix)<boundary:
                raise EngineError('TOKEN_ALIGNMENT_FAILED','Chat-template prefix changes across role boundaries')
            roles.append(span(message['role'],boundary,len(prefix),'complete_message_envelope'))
            boundary=len(prefix)
        mapping='complete_chat_template_envelopes'
    else:
        # Injectable test tokenizers may not implement templates; never label content-only as full.
        for message in rendered_messages:
            start=rendered.find(message['content'],boundary)
            if start<0:raise EngineError('TOKEN_ALIGNMENT_FAILED','Role content absent from rendered text')
            end=start+len(message['content']);roles.append(span(message['role'],start,end,'message_content_only'));boundary=end
        mapping='content_only_test_tokenizer'
    if boundary<len(rendered):roles.append(span('assistant',boundary,len(rendered),'generation_prompt'))
    if continuation:roles.append(span('assistant',len(rendered),len(text),'fixed_continuation'))
    return {'special_token_mask':special_mask,'special_token_mask_unavailable_reason':None if special_mask is not None else 'tokenizer_special_ids_unavailable','token_character_offsets':[list(pair) for pair in offsets],'role_spans':roles,'role_span_mapping':mapping}
