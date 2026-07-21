"""LLM-as-judge: pointwise, reference-free faithfulness verdicts.

Reads an extraction bundle, judges each extracted record against the paper's full text
(one call per record: rubric + paper + record -> critique, bad_fields, verdict), and
writes a judge bundle: config.json (content-hashed), verdicts/<doi>.json, judge_meta.json.
The judge never sees the curated data or the metric's verdicts."""
import json
from pathlib import Path
from typing import Literal

import weave
import litellm
from tqdm import tqdm

from pydantic import BaseModel, Field, ValidationError

from .utils import filename_to_doi, doi_to_filename
from . import bundle, tracking
from .paths import prompt_path, data_path, RUNS_DIR


class JudgeVerdict(BaseModel):
    # critique comes first on purpose: it is generated before the verdict, so the model
    # must reason before it commits (structured-output "think step by step").
    critique: str = Field(description="2-5 sentences citing the specific evidence")
    bad_fields: list[str] = Field(default_factory=list,
                                  description="Names of fields judged wrong/unsupported/missing")
    verdict: Literal["correct", "incorrect"]


def read_rubric(harness_params: dict) -> str:
    return prompt_path(harness_params["rubric_file"]).read_text(encoding="utf-8")


def construct_messages(rubric: str, full_text: str, record: dict) -> list[dict]:
    return [{"role": "system", "content": rubric},
            {"role": "user",
             "content": f"PAPER TEXT:\n{full_text}\n\nEXTRACTED RECORD:\n"
                        f"{json.dumps(record, indent=2)}"}]


@weave.op(postprocess_output=lambda out: {"verdict": out[0].model_dump() if out and out[0] else None})
def run_llm(llm_params: dict, messages, **kwargs):
    try:
        resp = litellm.completion(messages=messages, response_format=JudgeVerdict,
                                  num_retries=5, **llm_params, **kwargs)
    except litellm.AuthenticationError as e:
        raise RuntimeError(f"Authentication error: {e}. Check your API key.")
    except litellm.RateLimitError as e:
        raise RuntimeError(f"Rate limit / quota error: {e}. Try again later or switch judge model.")
    except litellm.APIError as e:
        raise RuntimeError(f"API error: {e}. LLM service issue.")
    content = resp.choices[0].message.content
    # Some endpoint models ignore response_format (return prose/fenced JSON): salvage the
    # outermost {...} before giving up.
    for candidate in (content, content[content.find("{"): content.rfind("}") + 1]):
        try:
            return JudgeVerdict.model_validate_json(candidate), resp, True
        except (ValidationError, ValueError):
            continue
    return None, resp, False


def run(env: dict, run_dir: Path, limit: int | None = None) -> None:
    hp = env["harness_params"]
    rubric = read_rubric(hp)
    ext_dir = RUNS_DIR / hp["extraction_run"]
    md_dir = data_path(hp.get("curated_data_markdown_dir", "curated_data_markdown_by_doi"))

    config = bundle.unpack_config(env)
    config["harness_params"]["rubric"] = rubric              # embed actual text, like prompts
    bundle.write_json(run_dir / "config.json",
                      {"content_hash": bundle.content_hash(config), **config})

    ext_files = sorted((ext_dir / "extractions").glob("*.json"))
    if limit:
        ext_files = ext_files[:limit]

    meta = {"model": env["llm_params"]["model"], "extraction_run": hp["extraction_run"],
            "git_commit": bundle.git_commit(), "started_at": bundle.now_iso(),
            "n_papers": len(ext_files), "n_records": 0, "parse_failed_records": 0,
            "prompt_tokens": 0, "completion_tokens": 0}

    for f in tqdm(ext_files, desc="judge"):
        d = bundle.read_json(f)
        doi = d["doi"]
        full_text = (md_dir / doi_to_filename(doi, "md")).read_text(encoding="utf-8")
        verdicts = []
        for i, record in enumerate(d["records"]):
            v, resp, parsed_ok = run_llm(env["llm_params"],
                                         construct_messages(rubric, full_text, record))
            if not parsed_ok:
                meta["parse_failed_records"] += 1
                bundle.write_json(run_dir / "raw" / f"{f.stem}_{i}.json",
                                  {"doi": doi, "extracted_index": i,
                                   "response_content": resp.choices[0].message.content})
            verdicts.append({"extracted_index": i, "parsed_ok": parsed_ok,
                             **(v.model_dump() if v else {})})
            usage = resp.usage
            meta["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            meta["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        meta["n_records"] += len(d["records"])
        bundle.write_json(run_dir / "verdicts" / f.name, {"doi": doi, "verdicts": verdicts})

    meta["finished_at"] = bundle.now_iso()
    bundle.write_json(run_dir / "judge_meta.json", meta)
    print(f"judge bundle -> {run_dir}  ({meta['n_records']} records, "
          f"{meta['parse_failed_records']} unparseable)")
    tracking.log_bundle(run_dir, stage="judge")
