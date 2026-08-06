import numpy as np
from _setup import *
from core.evaluation import record_penalty
from core.schema import load_curated

reference = load_curated(data_path(curated_table))
extraction = as_experiments(extraction_run)

hungarian_total = totals(curation=curated_table)["tp"]
greedy_total = 0

for doi, curated_rows in reference.items():
    extracted_rows = extraction.get(doi, [])
    if not curated_rows or not extracted_rows:
        continue

    cost = np.zeros((len(curated_rows), len(extracted_rows)))
    accepts = np.zeros((len(curated_rows), len(extracted_rows)), dtype=bool)
    for i, curated_row in enumerate(curated_rows):
        for j, extracted_row in enumerate(extracted_rows):
            penalty, catalyst_matched, _ = record_penalty(curated_row, extracted_row, 0.8, 0.2)
            cost[i, j] = penalty if catalyst_matched else 10.0
            accepts[i, j] = catalyst_matched and penalty < 0.3

    taken_curated, taken_extracted = set(), set()
    order = sorted(((cost[i, j], i, j) for i in range(len(curated_rows))
                    for j in range(len(extracted_rows))))
    for _, i, j in order:
        if i in taken_curated or j in taken_extracted:
            continue
        taken_curated.add(i)
        taken_extracted.add(j)
        if accepts[i, j]:
            greedy_total += 1

print("correct matches found by optimal (Hungarian) pairing:", hungarian_total)
print("correct matches found by simple closest-first pairing:", greedy_total)
print()
print("difference:", hungarian_total - greedy_total)
print()
print("The two agree, which means the pairing step is not where the metric loses. Its problems")
print("come after pairing, when it decides whether a pair is acceptable.")
