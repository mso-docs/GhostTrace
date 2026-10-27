"""Metric v1: descriptive distances, never an emotional-state score."""
import math

def distances(a,b):
    if len(a)!=len(b) or not a:raise ValueError('Vectors must have equal nonzero length')
    if any(not math.isfinite(float(x)) for x in [*a,*b]):raise ValueError('Nonfinite activation')
    norm_a=math.sqrt(sum(float(x)**2 for x in a));norm_b=math.sqrt(sum(float(x)**2 for x in b))
    l2=math.sqrt(sum((float(x)-float(y))**2 for x,y in zip(a,b)))
    return {'normalized_l2':l2/(0.5*(norm_a+norm_b)+1e-8),'cosine_distance':None if min(norm_a,norm_b)<=1e-8 else max(0.,min(2.,1-sum(float(x)*float(y) for x,y in zip(a,b))/(norm_a*norm_b))),'cosine_missing_reason':'zero_norm' if min(norm_a,norm_b)<=1e-8 else None,'norm_ratio':norm_a/(norm_b+1e-8)}

def lexical_counts(text):
    import re
    vocabulary={'regret':['sorry','regret','apologize','apology'],'responsibility':['my fault','my responsibility','i caused','i should have'],'repair':['replace','repair','make it right','help'],'deflection':['not my fault','overreact','no choice','justified']}
    lowered=text.lower()
    return {'version':'lexical-v1','interpretation':'Weak descriptive phrase counts; not a remorse score','counts':{k:sum(len(re.findall(r'\b'+re.escape(p)+r'\b',lowered)) for p in words) for k,words in vocabulary.items()}}
