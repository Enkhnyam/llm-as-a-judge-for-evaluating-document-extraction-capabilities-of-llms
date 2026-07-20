import random
import json
from pathlib import Path

import weave
import litellm
from tqdm import tqdm

from pydantic import ValidationError

from .schema import ExtractionResponse
from .utils import filename_to_doi, doi_to_filename
from . import bundle, tracking
from .paths import resolve
from .licensing import licensable_dois

def _cost(resp) -> float:
    try:
        return float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:
        return float((getattr(resp, "_hidden_params", {}) or {}).get("response_cost") or 0.0)

@weave.op(postprocess_output=lambda out: {"records": out[0] if out else []})
def run_llm(llm_params: dict, messages, **kwargs):
    """Call the model and parse its JSON into records"""
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

    try:
        records = ExtractionResponse.model_validate_json(resp.choices[0].message.content)
        return records.experiments, resp, True
    except ValidationError:
        return [], resp, False


def read_prompt(harness_params: dict) -> str:
    return resolve(harness_params.get("prompt_file", "prompt.txt")).read_text(encoding="utf-8")

@weave.op()
def construct_prompt(harness_params: dict, target_doi: str) -> list[dict]:
    n_shots = harness_params["n_shots"]
    prompt = read_prompt(harness_params)
    rng = random.Random(harness_params["seed"])

    curated_json = resolve(harness_params.get("curated_data_path", "curated_data_json_by_doi.json"))
    data = json.loads(curated_json.read_text(encoding="utf-8"))
    target_paper = next(x for x in data if x["doi"] == target_doi)
    # Few-shot exemplars must be redistributable: draw only from the licensable papers
    # (this is why n_shots caps at 6 — 7 licensable papers minus the held-out target).
    allowed = licensable_dois()
    example_papers = [x for x in data if x["doi"] != target_doi and x["doi"] in allowed]

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

def run(env: dict, run_dir: Path, limit: int | None = None) -> None:
    md_dir = resolve(env["harness_params"].get("curated_data_markdown_dir", "curated_data_markdown_by_doi"))
    md_files = sorted(md_dir.glob("*.md")) 
    if limit:
        md_files = md_files[:limit]

    config = bundle.unpack_config(env)
    config["harness_params"]["prompt"] = read_prompt(env["harness_params"])  # embed actual text
    bundle.write_json(run_dir / "config.json",
                      {"content_hash": bundle.content_hash(config), **config})

    meta = {"seed": env["harness_params"]["seed"], "model": env["llm_params"]["model"],
            "git_commit": bundle.git_commit(), "started_at": bundle.now_iso(),
            "n_papers": len(md_files), "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0.0, "parse_failed_papers": 0}

    for md in tqdm(md_files, desc="extract"):
        doi = filename_to_doi(md.name)
        messages = construct_prompt(env["harness_params"], doi)
        records, resp, parsed_ok = run_llm(env["llm_params"], messages)
        if not parsed_ok:
            meta["parse_failed_papers"] += 1
        fn = doi_to_filename(doi, filetype="json")
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
    if meta["parse_failed_papers"]:
        print(f"  note: {meta['parse_failed_papers']}/{meta['n_papers']} papers returned "
              f"unparseable output (0 records); see raw/*.json 'response_content'.")
    tracking.log_bundle(run_dir, stage="extract")
