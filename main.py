import litellm
from pathlib import Path
from envyaml import EnvYAML
import os, json

current_dir = Path(__file__).parent

config_file_path = current_dir / "first_bundle_config.yaml"

env_file_path = current_dir / ".env"

curated_data_json_path = current_dir / "curated_data_json_by_doi.json"

def track_cost(kwargs, completion_response, start_time, end_time):
    print("Cost:", kwargs.get("response_cost", 0))

def run_llm(model_params: EnvYAML, messages, tools=None, **kwargs):
    litellm.success_callback = [track_cost, *model_params["success_callback"]]
    try:
        resp = litellm.completion(
            model=model_params["model_name"],
            messages=messages,
            tools=tools,
            **model_params["params"], **kwargs
        )
    except litellm.AuthenticationError as e:
        print(f"Bad API key: {e}")
    except litellm.RateLimitError as e:
        print(f"Rate limited: {e}")
    except litellm.APIError as e:
        print(f"API error: {e}")

    return resp

def construct_prompt(params: EnvYAML):
    messages = []
    n_shots = params.get("params", {}).get("n_shots", 0)
    prompt = params.get("prompt", "")
    with open(curated_data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for paper in data[:n_shots]:
            messages.append(
                {
                    "role": "user",
                    "content": f"Paper DOI: {paper['doi']}\nTitle: {paper['title']}\nFull Text:\n{paper['full_text']}\n\n{prompt}"
                }
            )
            messages.append({
                "role": "assistant",
                "content": {json.dumps(paper['extracted_experiments'], indent=4)}
            })

    return messages

def main():
    env = EnvYAML(config_file_path, env_file=env_file_path)
    messages = construct_prompt(env)
    resp = run_llm(env, messages, tools=None)
    


if __name__ == "__main__":
    main()
