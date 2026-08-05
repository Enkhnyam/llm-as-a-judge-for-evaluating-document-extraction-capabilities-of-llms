"""Is the judge really better than the metric, or is it chance?

Both graders scored the same records, so the comparison is paired: count the records where
exactly one of them matched the chemists, and ask how surprising that split is.
"""
from scipy.stats import binomtest
from _setup import *

labelled = golden()
splits = {"all": labelled, "test (held out)": labelled[labelled.split == "test"]}

comparison = {}
for split_name, part in splits.items():
    judge_alone = ((part.judge_v4 == part.human) & (part.metric != part.human)).sum()
    metric_alone = ((part.metric == part.human) & (part.judge_v4 != part.human)).sum()
    p_value = binomtest(judge_alone, judge_alone + metric_alone).pvalue
    comparison[split_name] = {"records": len(part),
                              "only the judge right": judge_alone,
                              "only the metric right": metric_alone,
                              "p": p_value}

print(pd.DataFrame(comparison).T.to_string(float_format="{:.4f}".format))
print()
print("The test split was fixed before the final rubric was written, so it is held out.")
