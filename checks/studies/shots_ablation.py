"""Does giving the model more worked examples improve extraction?"""
from _setup import *

models = {"gpt-oss-120b": "shots_oss", "mistral-small": "shots_mistral"}

for model_name, folder in models.items():
    results = scores_by_run(folder)
    if results.empty:
        print(model_name + ": not run yet")
        print()
        continue
    by_shots = results.groupby("n_shots").f1.agg(["mean", "std", "count"])
    print(model_name)
    print(by_shots.to_string(float_format="{:.3f}".format))
    print()
