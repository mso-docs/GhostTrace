import random
import statistics

def bootstrap_scenarios(values,draws=2000,seed=20261027):
    """Input contains ONE pre-averaged paired distance per scenario."""
    if not values:return {'mean':None,'lower':None,'upper':None,'n':0,'reason':'no_valid_matched_scenarios'}
    rng=random.Random(seed);n=len(values)
    means=sorted(statistics.fmean(rng.choices(values,k=n)) for _ in range(draws))
    def percentile(p):
        x=(len(means)-1)*p;i=int(x);return means[i]+(means[min(i+1,len(means)-1)]-means[i])*(x-i)
    return {'mean':statistics.fmean(values),'lower':percentile(.025),'upper':percentile(.975),'n':n,'bootstrap_draws':draws,'bootstrap_seed':seed,'unit':'scenario','caveat':'Descriptive pilot interval assumes exchangeable scenarios; family dependence and small samples limit generalization.'}
