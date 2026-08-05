"""Precision, recall and F1 of the extraction against each curated dataset.

The original curation is what the graders were run against, so it is the reference behind the
labelled set. The frozen curation adds experiments that were missing from it, so it scores the
same extraction more fairly. Both are reported because they answer different questions.
"""
from _setup import *

records = extracted()
against_original = totals(curation=original)
against_frozen = totals(curation=frozen)

comparison = pd.DataFrame([against_original, against_frozen],
                          index=["original curation", "frozen curation"])
comparison = comparison[["precision", "recall", "f1", "tp", "fp", "fn"]]

print("run     ", extraction_run)
print("records ", len(records), "extracted from", records.doi.nunique(), "papers")
print()
print(comparison.to_string(float_format="{:.3f}".format))
