"""Does the curated data obey yield = conversion x selectivity / 100?

The judge forgives a missing outcome when the other two imply it, which is only fair if the
relation holds in the data. This measures it instead of assuming it.
"""
from _setup import *

experiments = curated()
complete = experiments.dropna(subset=outcomes)

implied_yield = complete.conversion_percent * complete.selectivity_percent / 100
error = (complete.yield_percent - implied_yield).abs()
summary = error.describe().to_string(float_format="{:.3f}".format)

print("experiments reporting all three outcomes", len(complete), "of", len(experiments))
print()
print("difference between the reported and the implied yield:")
print(summary)
