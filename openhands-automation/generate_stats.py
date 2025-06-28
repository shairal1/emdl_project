import json
import logging
import time
from openai import OpenAI
from pathlib import Path
from typing import Dict, List

PROMPT = """
You are given five code blocks:

1. **##ISSUE_NAME##** - A short identifier that tells you exactly which kind of problem to focus on (e.g., “spurious_correlations”, “data_leakage”, “no_cross_validation”).  
2. **##ORIGINAL_CODE##** - The original code that may contain this problem.  
3. **##FIXED_CODE##** - The correct version manually fixed by a human; treat this as ground-truth.  
4. **##AI_FIXED_CODE##** - The version produced by an AI agent, which includes a summary of its changes at the top.  
5. **##EXPLANATION##** - A brief explanation of the actual issue, for your reference.

---

### Your task  
Compare **AI_FIXED_CODE** against **FIXED_CODE** (using **ORIGINAL_CODE** for context) and decide, *with respect only to the issue named in **##ISSUE_NAME##***:

1. **issue_detected** - Did the AI’s summary clearly acknowledge the named issue and attempt to address it?  
2. **issue_fixed** - Did the AI’s code changes actually solve the named issue in the same way (or an equally valid way) as the human fix?  
3. **false_positive** - Did the AI modify code that was already correct but unrelated to the named issue? Only set to **true** if those changes affect the ML pipeline itself and did not help resolve the named issue; ignore any edits to imports, logging, script structure, formatting, or other non-pipeline aspects; always **false** if the named issue was correctly fixed.

Ignore all other differences (logging, paths, formatting, etc.) that are not directly related to **##ISSUE_NAME##**.

---

### Output  
Return **only** the following JSON object – no extra text:

{
  "issue_detected": boolean,         // true ⇢ AI acknowledged & targeted the named issue (or fixed it without mentioning)
  "issue_fixed": boolean,            // true ⇢ AI’s changes resolve the issue as well as the human fix
  "false_positive": boolean,         // true ⇢ AI introduced changes to the ML pipeline outside the scope of the named issue and did not help resolve it; ignore any edits to imports, logging, script structure, formatting, or other non-pipeline aspects; always false if the named issue was correctly fixed.
  "description": {
    "issue_detection": "short reason for true/false",
    "issue_fix": "short reason for true/false",
    "false_positive": "short reason for true/false"
  }
}

Only return the JSON object — no additional text or explanation.
"""


def gather_examples_with_fixed_code(root: Path) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for script in root.glob("pipelines/*/example-0.py"):
        try:
            ai_fixed_path = script.parent / "fixed.py"
            fixed_path = script.parent / "example-0-fixed.py"
            explation_path = script.parent / "example-0-explanation.md"

            if not ai_fixed_path.exists():
                logging.info("Skipping %s, fixed.py not exists.", script.relative_to(root))
                continue

            if not fixed_path.exists():
                logging.info("Skipping %s, example-0-fixed.py not exists.", script.relative_to(root))
                continue

            if not explation_path.exists():
                logging.info("Skipping %s, example-0-explanation.md not exists.", script.relative_to(root))
                continue

            content = script.read_text(encoding="utf-8")
            ai_fixed = ai_fixed_path.read_text(encoding="utf-8")
            fixed =  fixed_path.read_text(encoding="utf-8")
            explanation =  explation_path.read_text(encoding="utf-8")
            

            examples.append(
                {
                    "name": script.parent.name,
                    "path": script.parent,
                    "content": content,
                    "ai_fixed_content": ai_fixed,
                    "fixed_content": fixed,
                    "explanation": explanation
                }
            )
            logging.info("Loaded %s", script.relative_to(root))
        except Exception:
            logging.exception("Unable to read %s", script)
    return examples

def process_example_with_openai(example: Dict[str, str], client: OpenAI) -> Dict[str, any]:
    try:
        prompt = PROMPT.replace("##ISSUE_NAME##", example["name"])
        prompt = prompt.replace("##ORIGINAL_CODE##", example["content"])
        prompt = prompt.replace("##FIXED_CODE##", example["fixed_content"])
        prompt = prompt.replace("##AI_FIXED_CODE##", example["ai_fixed_content"])
        prompt = prompt.replace("##EXPLANATION##", example["explanation"])

        tokens = int(len(prompt.split()) * 1.5)
        logging.info("Processing %s with %d tokens", example["name"], tokens)
        
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a code analysis expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        logging.info("Successfully processed %s", example["name"])
        return result
        
    except Exception as e:
        logging.exception("Failed to process %s: %s", example["name"], str(e))
        return {
            "issue_detected": False,
            "issue_fixed": False,
            "false_positive": False,
            "description": {
                "issue_detection": "Error processing",
                "issue_fix": "Error processing",
                "false_positive": "Error processing"
            },
            "error": str(e)
        }

def main():
    """Main function to process all examples and generate statistics."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    client = OpenAI()  # OPENAI_API_KEY is set in environment
    
    root = Path(__file__).parent
    
    examples = gather_examples_with_fixed_code(root)
    logging.info("Found %d examples to process", len(examples))
    
    stats = {}
    
    for example in examples:
        logging.info("Processing %s...", example["name"])
        result = process_example_with_openai(example, client)
        stats[example["name"]] = result
        
        time.sleep(1)
    
    output_file = root / "stats.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logging.info("Statistics saved to %s", output_file)
    
    total = len(stats)
    issues_detected = sum(1 for stat in stats.values() if stat.get("issue_detected", False))
    issues_fixed = sum(1 for stat in stats.values() if stat.get("issue_fixed", False))
    false_positives = sum(1 for stat in stats.values() if stat.get("false_positive", False))
    errors = sum(1 for stat in stats.values() if "error" in stat)
    
    print(f"\n--- Summary ---")
    print(f"Total examples processed: {total}")
    print(f"Issues detected: {issues_detected}")
    print(f"Issues fixed: {issues_fixed}")
    print(f"False positives: {false_positives}")
    print(f"Processing errors: {errors}")
    print(f"Statistics saved to: {output_file}")

if __name__ == "__main__":
    main()