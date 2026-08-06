from _setup import *
from _curation import *

raw = json.loads(data_path(as_curated).read_text())
started_with = sum(len(paper["extracted_experiments"]) for paper in raw)

papers = cleaned_papers()
removed = started_with - sum(len(paper["extracted_experiments"]) for paper in papers)

by_doi = {paper["doi"]: paper for paper in papers}
records = extracted_records()
candidates = missing_experiments()
verified = candidates[candidates.human == "correct"]

added = 0
for row in verified.itertuples():
    experiment = records[(row.doi, row.extracted_index)]
    fingerprint = tuple(str(experiment.get(field)) for field in DATA_FIELDS)
    existing = {tuple(str(entry["experiment_data"].get(field)) for field in DATA_FIELDS)
                for entry in by_doi[row.doi]["extracted_experiments"]}
    if fingerprint in existing:
        continue
    by_doi[row.doi]["extracted_experiments"].append({"experiment_data": experiment})
    added += 1

total = write_version(curated_table, papers)
unverified = len(candidates) - len(verified)

print("hand-curated experiments        ", started_with)
print("entered twice or no outcome     ", -removed)
print("missing experiments added       ", added)
print("curated table                   ", total)
print()
print(unverified, "further candidates are not in the table.")
