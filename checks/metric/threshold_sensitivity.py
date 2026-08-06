from _setup import *

def sweep(setting, values):
    results = {}
    for value in values:
        results[value] = totals(curation=curated_table, **{setting: value})
    table = pd.DataFrame(results).T
    table = table[["f1", "precision", "recall"]]
    table.index.name = setting
    return table.to_string(float_format="{:.3f}".format)

print("acceptance cutoff (default 0.30)")
print(sweep("accept", [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]))
print()
print("catalyst name similarity required (default 0.80)")
print(sweep("catalyst", [0.6, 0.7, 0.8, 0.9, 1.0]))
print()
print("numeric tolerance (default 0.20)")
print(sweep("tolerance", [0.05, 0.1, 0.2, 0.3, 0.5]))
