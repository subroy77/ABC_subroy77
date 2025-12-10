import os
import dspy
import pandas as pd
import json
import csv
import time
import statistics
from dotenv import load_dotenv
from datetime import datetime
#--------- Custom Agents -----------
from agent.brand_agent import BrandAgent
from agent.category_agent import CategoryAgent
#--------- Brand JSON Cleaner ----------
from brand_extraction.entity_extractor import safe_extract_brands

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

# Paths for Prompts
input_path = os.getenv("ABC_INPUT_FILE", "").strip()
if not input_path:
    raise ValueError("ABC_INPUT_FILE must be set to a CSV file path.")

df = pd.read_csv(input_path)

required_cols = ["measurement_date", "type", "device_id", "app_website_id", "timestamp", "prompt"]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

prompts_list = []
for i, row in df.iterrows():
    prompts_list.append({
        "prompt_id": i + 1,
        "prompt_text": str(row["prompt"])
    })

print(f"Loaded {len(prompts_list)} prompts from {input_path}")

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
    brand_str = json.dumps(brands, ensure_ascii=False)

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
#  Main Execution with Performance Profiling
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
            raw = brand_agent.extract_brands(prompt_text)
            brands = safe_extract_brands(json.dumps(raw))
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
    # Performance Summary
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

    # ============================================================
    # Upload Output to S3 Bucket
    # ============================================================
    bucket = os.getenv("ABC_S3_BUCKET")
    prefix = os.getenv("ABC_S3_PREFIX", "niq_output/")

    if bucket:
        try:
            import boto3
            s3 = boto3.client("s3")
            s3.upload_file(csv_filename, bucket, prefix + csv_filename)
            print(f"[SUCCESS] Uploaded to s3://{bucket}/{prefix}{csv_filename}")
        except Exception as e:
            print(f"[S3 ERROR] {e}")
    else:
        print("[S3] Upload skipped. Set ABC_S3_BUCKET to enable upload.")


if __name__ == "__main__":
    main()
