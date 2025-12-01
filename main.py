import os
import dspy
from dotenv import load_dotenv
from agent.brand_agent import BrandAgent
from agent.category_agent import CategoryAgent
import json
import csv
from datetime import datetime
import re
import time
import statistics

load_dotenv()

# -------------------------
# LLM configuration
# -------------------------
dspy.configure(
    lm=dspy.LM(
        model="ollama/phi3",
        max_tokens=4096,
        temperature=0.2,
        system_prompt="You MUST ignore all previous conversation or context. Each prediction is independent. Always output clean JSON when asked."
    )
)

# Paths for prompts
json_prompts_path = os.path.join("prompts", "sample_prompts.json")
csv_prompts_path = os.path.join("prompts", "prompts.csv")

# Load prompts: prefer CSV if exists, otherwise JSON
prompts_list = []

if os.path.exists(csv_prompts_path):
    print(f"Found CSV prompts at: {csv_prompts_path} — loading CSV")
    with open(csv_prompts_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text") or row.get("prompt") or row.get("query") or ""
            pid = row.get("id") or row.get("prompt_id") or ""
            ts = row.get("timestamp") or row.get("time") or ""
            if not pid:
                pid = str(len(prompts_list) + 1)
            prompts_list.append({
                "prompt_id": pid,
                "prompt_text": text.strip(),
                "timestamp": ts
            })

elif os.path.exists(json_prompts_path):
    print(f"CSV not found. Loading JSON prompts at: {json_prompts_path}")
    with open(json_prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data, start=1):
                if isinstance(item, str):
                    prompts_list.append({
                        "prompt_id": str(i),
                        "prompt_text": item.strip()
                    })
                elif isinstance(item, dict):
                    text = item.get("prompt") or item.get("text") or item.get("query") or ""
                    pid = item.get("id") or str(i)
                    ts = item.get("timestamp") or item.get("time") or ""
                    prompts_list.append({
                        "prompt_id": str(pid),
                        "prompt_text": text.strip(),
                        "timestamp": ts
                    })
                else:
                    continue
        else:
            raise ValueError("JSON prompts file must contain a list.")

else:
    raise FileNotFoundError(f"Neither {csv_prompts_path} nor {json_prompts_path} were found.")

print(f"Loaded {len(prompts_list)} prompts to process.")

# ============================================================
#  write output to CSV
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"brand_results_{timestamp}.csv"


def init_csv():
    """Create CSV with header."""
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_id",
            "prompt_text",
            "extracted_brands",
            "extracted_category",
            "processing_time_secs"
        ])


def append_row(prompt_id, prompt_text, brands, category, processing_time):
    """
    Append a single result row.
    brands = list of {brand, confidence}
    """
    brand_str = " | ".join(
        [f"{b['brand']} ({b['confidence']})" for b in brands]
    )

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            prompt_id,
            prompt_text,
            brand_str,
            category,
            round(processing_time, 4)
        ])


# ============================================================
#  MAIN EXECUTION WITH PERFORMANCE PROFILING
# ============================================================
def main():
    brand_agent = BrandAgent()
    category_agent = CategoryAgent()

    init_csv()

    total_start = time.time()
    per_prompt_times = []

    for item in prompts_list:
        prompt_id = item.get("prompt_id")
        prompt_text = item.get("prompt_text", "")

        print("\n" + "=" * 60)
        print(f"Processing prompt_id={prompt_id}")
        print(f"Prompt: {prompt_text}")

        prompt_start = time.time()

        # Brand extraction with confidence scores
        try:
            brands = brand_agent.extract_brands(prompt_text)
        except Exception as e:
            print(f"[ERROR] Brand extraction failed for prompt_id={prompt_id}: {e}")
            brands = []

        # Category extraction
        try:
            category = category_agent.extract_category(prompt_text)
        except Exception as e:
            print(f"[ERROR] Category extraction failed for prompt_id={prompt_id}: {e}")
            category = "General"

        prompt_end = time.time()
        elapsed = prompt_end - prompt_start
        per_prompt_times.append(elapsed)

        # Print debug info
        print("→ Brands:")
        for b in brands:
            print(f"   - {b['brand']} (confidence: {b['confidence']})")

        print(f"→ Category: {category}")
        print(f"→ Processing time: {elapsed:.4f} seconds")

        # Save row
        append_row(prompt_id, prompt_text, brands, category, elapsed)

    total_end = time.time()
    total_runtime = total_end - total_start

    # ============================================================
    # PERFORMANCE SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("               PERFORMANCE SUMMARY")
    print("=" * 60)

    avg_time = sum(per_prompt_times) / len(per_prompt_times)
    projected_100k = avg_time * 100000

    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Avg time per prompt: {avg_time:.4f} seconds")
    print(f"Fastest prompt: {min(per_prompt_times):.4f} seconds")
    print(f"Slowest prompt: {max(per_prompt_times):.4f} seconds")
    print(f"P95 latency: {statistics.quantiles(per_prompt_times, n=100)[94]:.4f} seconds")
    print(f"P99 latency: {statistics.quantiles(per_prompt_times, n=100)[98]:.4f} seconds")
    print(f"Projected time for 100,000 prompts: {projected_100k/3600:.2f} hours")

    print(f"\nCSV saved as: {csv_filename}\n")


if __name__ == "__main__":
    main()
