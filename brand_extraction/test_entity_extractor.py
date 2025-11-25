import json
from entity_extractor import extract_entities

with open("data/golden_data_use_case_2 1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
for example in data["examples"]:
    prompt = example["prompt"]
    extraction = extract_entities(prompt)
    result = {
        "prompt": prompt,
        "expected_brands": example["expected_brands"],
        "category": example["category"],
        "entity_extractor_output": extraction
    }
    results.append(result)

with open("data/entity_extractor_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Results written to data/entity_extractor_output.json")
