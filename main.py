import json
import random
import argparse
from pathlib import Path

import weave
import litellm
from envyaml import EnvYAML
from tqdm import tqdm

import bundle
import parse
from evaluation import evaluate
from schema import Experiment, ExtractionResponse, load_curated
from utils import filename_to_doi

current_dir = Path(__file__).parent
env_file_path = current_dir / ".env"
curated_data_json_path = current_dir / "curated_data_json_by_doi.json"
curated_data_markdown_dir = current_dir / "curated_data_markdown_by_doi"
runs_dir = current_dir / "runs"


def track_cost(kwargs, completion_response, start_time, end_time):
    print("Cost:", kwargs.get("response_cost", 0))


@weave.op(postprocess_output=lambda out: {"records": out[0]})
def run_llm(llm_params: dict, messages, **kwargs):
    try:
        resp = litellm.completion(
            messages=messages,
            response_format=ExtractionResponse,
            num_retries=5,
            **llm_params, **kwargs,
        )
    except litellm.AuthenticationError as e:
        raise RuntimeError(f"Authentication error: {e}. Check your API key.")
    except litellm.RateLimitError as e:
        raise RuntimeError(f"Rate limit error: {e}. Try again later.")
    except litellm.APIError as e:
        raise RuntimeError(f"API error: {e}. LLM service issue.")

    records = ExtractionResponse.model_validate_json(resp.choices[0].message.content)
    return records.experiments, resp


def read_prompt(harness_params: dict) -> str:
    return (current_dir / harness_params.get("prompt_file", "prompt.txt")).read_text(encoding="utf-8")

@weave.op()
def construct_prompt(harness_params: dict, target_doi: str) -> list[dict]:
    n_shots = harness_params["n_shots"]
    prompt = read_prompt(harness_params)
    rng = random.Random(harness_params["seed"])

    data = json.loads(curated_data_json_path.read_text(encoding="utf-8"))
    target_paper = next(x for x in data if x["doi"] == target_doi)
    example_papers = [x for x in data if x["doi"] != target_doi]

    def user_msg(paper):
        return {"role": "user",
                "content": f"Paper DOI: {paper['doi']}\nTitle: {paper['title']}\n"
                           f"Full Text:\n{paper['full_text']}\n\n{prompt}"}

    messages: list[dict] = []
    for paper in rng.sample(example_papers, min(n_shots, len(example_papers))):
        demo = [exp["experiment_data"] for exp in paper["extracted_experiments"]]
        messages.append(user_msg(paper))
        messages.append({"role": "assistant",
                         "content": json.dumps({"experiments": demo}, indent=2)})
    messages.append(user_msg(target_paper))
    return messages


def _cost(resp) -> float:
    try:
        return float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:
        return float((getattr(resp, "_hidden_params", {}) or {}).get("response_cost") or 0.0)
    

def cmd_parse(src: str, dest: str) -> None:
    if not src or not dest:
        raise ValueError("Both --src and --dest must be provided for parsing stage.")
    parse.parse_src_dest_directories(Path(src), Path(dest))


def cmd_extract(env, run_dir: Path, limit: int | None = None) -> None:
    md_files = sorted(curated_data_markdown_dir.glob("*.md"))
    if limit:
        md_files = md_files[:limit]

    config = bundle.unpack_config(env)
    config["harness_params"]["prompt"] = read_prompt(env["harness_params"])  # embed actual text
    bundle.write_json(run_dir / "config.json",
                      {"content_hash": bundle.content_hash(config), **config})

    meta = {"seed": env["harness_params"]["seed"], "model": env["llm_params"]["model"],
            "git_commit": bundle.git_commit(), "started_at": bundle.now_iso(),
            "n_papers": len(md_files), "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0.0}

    for md in tqdm(md_files, desc="extract"):
        doi = filename_to_doi(md.name)
        messages = construct_prompt(env["harness_params"], doi)
        records, resp = run_llm(env["llm_params"], messages)
        fn = bundle.doi_to_filename(doi)
        bundle.write_json(run_dir / "extractions" / fn,
                          {"doi": doi, "records": [r.model_dump(by_alias=True) for r in records]})
        bundle.write_json(run_dir / "raw" / fn,
                          {"doi": doi, "messages": messages,
                           "response_content": resp.choices[0].message.content})
        usage = resp.usage
        meta["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        meta["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        meta["cost_usd"] += _cost(resp)

    meta["finished_at"] = bundle.now_iso()
    bundle.write_json(run_dir / "run_meta.json", meta)
    print(f"extraction bundle -> {run_dir}  (${meta['cost_usd']:.4f})")


def cmd_eval(env, run_dir: Path) -> None:
    ev = env["harness_params"]["evaluation"]
    curated = load_curated(curated_data_json_path)

    extracted_by_doi: dict[str, list[Experiment]] = {}
    for f in sorted((run_dir / "extractions").glob("*.json")):
        d = bundle.read_json(f)
        extracted_by_doi[d["doi"]] = [Experiment.model_validate(r) for r in d["records"]]

    result, labels = evaluate(
        curated, extracted_by_doi,
        tp_threshold=ev["tp_threshold"],
        catalyst_threshold=ev["catalyst_threshold"],
        numeric_tolerance=ev["numeric_tolerance"],
    )
    bundle.write_json(run_dir / "eval.json", result)
    bundle.write_json(run_dir / "labels.json", labels)
    print(f"eval -> P={result['precision']:.3f} R={result['recall']:.3f} "
          f"F1={result['f1']:.3f}  (TP={result['tp']} FP={result['fp']} FN={result['fn']})")


def main():
    parser = argparse.ArgumentParser(prog="llm-as-a-judge-for-evaluating-document-extraction-capabilities-of-llms")
    parser.add_argument("stage", choices=["extract", "eval", "parse", "all"], default="all")
    parser.add_argument("--limit", type=int, nargs="?", default=None, help="Limit number of papers to process")
    parser.add_argument("--config", type=str, nargs="?", default="rwth_bundle_config.yaml", help="Path to bundle config YAML")
    parser.add_argument("--src", type=str, nargs="?", default=None, help="Source directory for parsing PDFs")
    parser.add_argument("--dest", type=str, nargs="?", default=None, help="Destination directory for parsed Markdown files")

    args = parser.parse_args()
    
    stage: str = args.stage
    limit: int = args.limit

    env = EnvYAML(current_dir / Path(args.config), env_file=env_file_path)
    litellm.success_callback = [track_cost, *env["success_callback"]]
    run_dir = runs_dir / env["run_name"]

    weave.init("llm-as-a-judge-for-evaluating-document-extraction-capabilities-of-llms")

    if stage in ("parse"):
        cmd_parse(args.src, args.dest)
    if stage in ("extract", "all"):
        cmd_extract(env, run_dir, limit)
    if stage in ("eval", "all"):
        cmd_eval(env, run_dir)


if __name__ == "__main__":
    main()
