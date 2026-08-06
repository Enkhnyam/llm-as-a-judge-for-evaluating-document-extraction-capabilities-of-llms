from _setup import *
from _curation import as_curated, curated_table

report = {}
for name, filename in [("as first curated", as_curated), ("corrected", curated_table)]:
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
ranges = curated(curated_table)[measured].agg(["min", "max"]).T

print(pd.DataFrame(report).to_string())
print()
print("value ranges:")
print(ranges.to_string())
