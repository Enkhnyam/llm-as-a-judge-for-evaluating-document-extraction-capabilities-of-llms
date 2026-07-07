import litellm
from pathlib import Path
from envyaml import EnvYAML
from utils import filename_to_doi
import json, glob, random
from tqdm import tqdm
from pydantic import BaseModel, Field

class Experiment(BaseModel):
    catalyst: str | None = Field(None, description="Catalyst name exactly as written")
    solvent: str | None = Field(None, description="Solvent name exactly as written")
    temperature_c: float | None = Field(None, description="Temperature in Celsius")
    reaction_time_min: float | None = Field(None, description="Reaction time in minutes")
    catalyst_amount_g: float | None = Field(None, description="Catalyst amount in grams")
    pet_amount_g: float | None = Field(None, description="PET amount in grams")
    solvent_amount_g: float | None = Field(None, description="Solvent amount in grams")
    yield_percent: float | None = Field(None, description="Yield %")
    selectivity_percent: float | None = Field(None, description="Selectivity %")
    conversion_percent: float | None = Field(None, description="Conversion %")
    pressure_atm: float | None = Field(None, description="Pressure in atm, only if stated")
    source_chunk_ids: list[str] = Field(default_factory=list,
        description="UUIDs (ID: tags) of the chunks these values came from")

class ExtractionResponse(BaseModel):
    experiments: list[Experiment] = Field(default_factory=list)


current_dir = Path(__file__).parent

config_file_path = current_dir / "first_bundle_config.yaml"

env_file_path = current_dir / ".env"

curated_data_json_path = current_dir / "curated_data_json_by_doi.json"

curated_data_markdown_dir = current_dir / "curated_data_markdown_by_doi"

def track_cost(kwargs, completion_response, start_time, end_time):
    print("Cost:", kwargs.get("response_cost", 0))

def run_llm(llm_params: dict, messages, tools=None, tool_choice=None, **kwargs):
    try:
        resp = litellm.completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            num_retries=5,
            **llm_params, **kwargs
        )
    except litellm.AuthenticationError as e:
        raise RuntimeError(f"Authentication error: {e}. Please check your API key and ensure it is valid.")
    except litellm.RateLimitError as e:
        raise RuntimeError(f"Rate limit error: {e}. You may have exceeded the allowed number of requests. Please try again later.")
    except litellm.APIError as e:
        raise RuntimeError(f"API error: {e}. There may be an issue with the LLM service. Please try again later.")

    tool_calls = resp.choices[0].message.tool_calls
    if not tool_calls:
        return []
    result = ExtractionResponse.model_validate_json(tool_calls[0].function.arguments)
    return result.experiments

def construct_prompt(harness_params: dict, target_doi: str):
    messages = []
    n_shots = harness_params["n_shots"]
    prompt = harness_params["prompt"]
    rng = random.Random(harness_params["seed"])
    with open(curated_data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        target_paper = next(x for x in data if x["doi"] == target_doi)
        example_papers = [x for x in data if x["doi"] != target_doi]
        for paper in rng.sample(example_papers, min(n_shots, len(example_papers))):
            messages.append(
                {
                    "role": "user",
                    "content": f"Paper DOI: {paper['doi']}\nTitle: {paper['title']}\nFull Text:\n{paper['full_text']}\n\n{prompt}"
                }
            )
            messages.append({
                "role": "assistant",
                "content": json.dumps(paper['extracted_experiments'], indent=4)
            })
        messages.append(
            {
                "role": "user",
                "content": f"Paper DOI: {target_paper['doi']}\nTitle: {target_paper['title']}\nFull Text:\n{target_paper['full_text']}\n\n{prompt}"
            }
        )

    return messages

def extraction_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "extract_experiments",
            "description": "Record all experiments extracted from this paper.",
            "parameters": ExtractionResponse.model_json_schema()
        }
    }
    

def main():
    env = EnvYAML(config_file_path, env_file=env_file_path)
    litellm.success_callback = [track_cost, *env["success_callback"]]

    for path in tqdm(glob.glob(str(curated_data_markdown_dir / "*.md"))):
        doi = filename_to_doi(Path(path).name)
        messages = construct_prompt(env["harness_params"], doi)
        records = run_llm(env["llm_params"], messages, tools=[extraction_tool()], tool_choice={"type": "function", "function": {"name": "extract_experiments"}})
        print(f"{doi}: {len(records)} experiments")


if __name__ == "__main__":
    main()
