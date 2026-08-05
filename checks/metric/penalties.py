"""Which fields the metric penalises, and how close accepted matches sit to the cutoff."""
from _setup import *

result = totals(curation=frozen)
field_errors = pd.Series(result["field_error_counts"]).sort_values(ascending=False)

labels = scored(curation=frozen)
accepted = labels[labels.verdict == "TP"]
penalties = accepted.avg_penalty.describe()

print("field disagreements among matched pairs:")
print(field_errors.to_string())
print()
print("penalty of accepted matches (a pair is accepted below 0.30):")
print(penalties.to_string(float_format="{:.3f}".format))
