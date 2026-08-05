"""Data-quality problems in the curated dataset, before and after cleaning.

The original curation is kept because it is the reference the graders were run against. The
frozen one is what scoring uses now: duplicated rows and rows with no measured outcome removed.
"""
from _setup import *

report = {}
for name, filename in [("original", original), ("frozen", frozen)]:
    experiments = curated(filename)
    percentages = experiments[outcomes]
    amounts = experiments[fields].select_dtypes("number")

    report[name] = {
        "experiments": len(experiments),
        "rows entered twice": experiments.duplicated(subset=["doi"] + fields).sum(),
        "rows with no outcome at all": percentages.isna().all(axis=1).sum(),
        "percentages outside 0-100": ((percentages < 0) | (percentages > 100)).sum().sum(),
        "negative amounts": (amounts < 0).sum().sum(),
        "rows citing no source chunk": experiments.source_chunk_ids.map(len).eq(0).sum(),
    }

measured = ["temperature_c", "reaction_time_min", "PET_amount_g", "catalyst_amount_g"]
ranges = curated(frozen)[measured].agg(["min", "max"]).T

print(pd.DataFrame(report).to_string())
print()
print("value ranges in the frozen curation:")
print(ranges.to_string())
