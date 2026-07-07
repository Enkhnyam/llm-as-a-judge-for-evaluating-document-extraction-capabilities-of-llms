from pathlib import Path
import json
from utils import filename_to_doi

current_dir = Path(__file__).parent
curated_data_json_path = current_dir / "extracted_data_with_sources.json"
curated_data_markdown_dir = current_dir / "curated_data_markdown_by_doi"

def main():
    markdown_papers = {}
    for md_path in curated_data_markdown_dir.glob("*.md"):
        doi = filename_to_doi(md_path.name)
        markdown_papers[doi] = md_path.read_text(encoding="utf-8")

    with open(curated_data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_objects = []

    for paper in data["papers"]:
        doi = paper["doi"]
        if doi not in markdown_papers:
            continue

        output_dict = {
            "doi": paper.get("doi"),
            "title": paper.get("paper_title"),
            "full_text": markdown_papers[doi],
            "extracted_experiments": [
                {
                    "experiment_number": ann.get("record_index", 0) + 1,
                    "experiment_data": ann.get("extracted_data"),
                }
                for ann in paper.get("annotations", [])
            ]
        }
        output_objects.append(output_dict)

    output_path = current_dir / "curated_data_json_by_doi.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_objects, f, indent=4)

if __name__ == "__main__":
    main()