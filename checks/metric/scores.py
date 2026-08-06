from _setup import *

records = extracted()
rows = {}
for name, filename in [("as first curated", as_curated), ("corrected", curated_table)]:
    result = totals(curation=filename)
    rows[name] = {"experiments in table": len(curated(filename)),
                  "precision": result["precision"], "recall": result["recall"],
                  "f1": result["f1"], "correct": result["tp"],
                  "false alarms": result["fp"], "missed": result["fn"]}

print("run     ", extraction_run)
print("records ", len(records), "extracted from", records.doi.nunique(), "papers")
print()
print(pd.DataFrame(rows).T.to_string(float_format="{:.3f}".format))
print()
print("Same extraction in both rows. The second corrects data-entry errors in the table and")
print("adds experiments it was missing.")
