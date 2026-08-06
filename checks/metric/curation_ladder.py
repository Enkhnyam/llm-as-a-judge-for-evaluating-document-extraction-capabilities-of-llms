from _setup import *
from _curation import as_curated, curated_table

rows = {}
for name, filename in [("as first curated", as_curated), ("corrected", curated_table)]:
    result = totals(curation=filename)
    rows[name] = {"experiments": len(curated(filename)), "precision": result["precision"],
                  "recall": result["recall"], "f1": result["f1"], "false alarms": result["fp"]}

print("the same extraction, scored against each table:")
print()
print(pd.DataFrame(rows).T.to_string(float_format="{:.3f}".format))
print()
print("Same extraction in both rows. Precision moves, recall does not: filling in missing")
print("experiments removes false alarms without creating misses.")
