from _setup import *

for model_name in ["oss", "mistral"]:
    citing = scores_by_run(f"src_{model_name}_on")
    not_citing = scores_by_run(f"src_{model_name}_off")
    if citing.empty or not_citing.empty:
        print(model_name + ": not run yet")
        print()
        continue

    comparison = pd.DataFrame({"citing sources": citing.f1, "not citing": not_citing.f1})
    summary = comparison.agg(["mean", "std", "count"]).T
    difference = citing.f1.mean() - not_citing.f1.mean()

    print(model_name)
    print(summary.to_string(float_format="{:.3f}".format))
    print(f"difference {difference:+.3f}")
    print()
