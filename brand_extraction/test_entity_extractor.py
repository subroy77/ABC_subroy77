import json
from entity_extractor import extract_entities

#--------testing safe_extract_brands with valid json & reasoning-------------
from entity_extractor import safe_extract_brands

def test_parse_valid_json():
    raw = '{"brands":[{"brand":"Samsung","confidence":1},{"brand":"LG","confidence":0.95}]}'
    assert len(safe_extract_brands(raw)) == 2

def test_parse_with_reasoning():
    raw = "<reasoning>hello</reasoning>{\"brands\":[{\"brand\":\"Sony\",\"confidence\":0.8}]}"
    result = safe_extract_brands(raw)
    assert result[0]["brand"] == "Sony"
#----------------------------------------------------------------------------

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

