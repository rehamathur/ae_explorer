"""
Process trial JSON files and generate CSV outputs for the Streamlit app.

This script takes an arbitrary list of trial JSON files and processes them through
the same pipeline as the notebook to generate:
- trials.csv
- ae.csv
- baseline.csv
- arms.csv
- trials_with_indication_groups.csv

Usage:
    # Command line:
    python process_trials.py trial1.json trial2.json --api-key YOUR_API_KEY
    
    # Or in Python:
    from process_trials import process_trials, load_trial_json_files
    
    trial_data = load_trial_json_files(["trial1.json", "trial2.json"])
    process_trials(
        trial_data=trial_data,
        openai_api_key="your-api-key",
        output_dir="./output"
    )
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple
from openai import OpenAI


def load_reducto_results(trial_data: List[dict]):
    """
    Process trial data from Reducto JSON format into DataFrames.
    
    Args:
        trial_data: List of trial objects, where each object is structured as:
            [{"trial_metadata": {...}, "baseline_table": [...], "ae_events": [...]}]
    
    Returns:
        df_trials, df_baseline, df_ae: Three DataFrames
    """
    trials = []
    baseline_rows = []
    ae_rows = []

    for i, trial_obj in enumerate(trial_data):
        trial_obj = trial_obj[0]
        md = trial_obj["trial_metadata"]

        # ---- trial_id / NCT handling ----
        nct = md["NCT Number"]
        
        trial_id = nct
        if trial_id == "N/A": 
            trial_id = f"{md['drug_name']}_{i+1}"
        print(trial_id)
        
        # ---- trials table row ----
        trial_row = {
            "trial_id": trial_id,
            "citation": md["citation"],
            "drug_name": md["drug_name"],
            "phase": md["phase"],
            "indication": md["indication"],
            "trial_duration": md["trial_duration"],
            "trial_duration_weeks": md["numeric_trial_duration"],
            "safety_population_definition": md["safety_population_definition"],
            "nct_number": nct or None,
            "sponsor": md["Sponsor"],
        }

        trials.append(trial_row)

        # ---- baseline table rows ----
        for row in trial_obj["baseline_table"]:
            arm = row["arm_result"] or {}
            baseline_rows.append({
                "trial_id": trial_id,
                "variable_label": row["variable_label"],
                "variable_label_raw": row["variable_label_raw"],
                "categorical_label": row["categorical_label"],
                "variable_type": row["variable_type"],
                "units": row["units"],

                "arm_label": arm["arm_label"],
                "arm_role": arm["arm_role"],
                "arm_total_n": arm["total_n"],

                "n": arm["n"],
                "percent": arm["percent"],
                "mean": arm["mean"],
                "sd": arm["sd"],
                "raw_value": arm["raw_value"],
            })

        # ---- AE table rows ----
        for row in trial_obj["ae_events"]:
            arm = row["arm_result"] or {}
            ae_rows.append({
                "trial_id": trial_id,
                "AE_label_raw": row["AE_label_raw"],
                "event_category": row["event_category"],
                "is_any_ae": row["is_any_ae"],
                "is_serious_ae": row["is_serious_ae"],
                "is_fatal": row["is_fatal"],
                "is_ae_leading_to_discontinuation": row["is_ae_leading_to_discontinuation"],
                "is_aesi": row["is_aesi"],

                "arm_label": arm["arm_label"],
                "arm_role": arm["arm_role"],
                "arm_total_n": arm["total_n"],
                "n_with_event": arm["n_with_event"],
                "percent_with_event": arm["percent_with_event"],
            })

    df_trials = pd.DataFrame(trials)
    df_baseline = pd.DataFrame(baseline_rows)
    df_ae = pd.DataFrame(ae_rows)

    return df_trials, df_baseline, df_ae


def build_prompt_for_labels(labels: List[str]) -> str:
    """Build prompt for harmonizing AE labels."""
    instructions = """
You are harmonizing adverse event labels from clinical trial safety tables.

Goal:
- Assign a canonical name for each label (ae_canonical) that merges variants of the same concept.
- Also assign a slightly broader "family" grouping (ae_family) that merges clearly related variants that may be slighlty clinically that someone would reasonably collapse for analysis when evaluating the AE profile of a drug.
- Do NOT merge clinically distinct events that are routinely reported separately (for example,
  "Nausea" vs "Vomiting" vs "Diarrhea" should each remain their own family) Remember these should be actually clinically distinct rathan a severe vs serious event. Imagine you are using these labels
  as a doctor to make treatment decisions. 

Definitions:
- ae_canonical:
  - A preferred term for the adverse event concept.
  - Examples:
    - "Cholecystitis acute" -> ae_canonical = "Acute cholecystitis"
    - "Chronic cholecystitis" -> ae_canonical = "Chronic cholecystitis"
    - "Cholelithiasis" -> ae_canonical = "Cholelithiasis"
    - "Nausea" -> ae_canonical = "Nausea"
    - "Renal event" -> ae_canonical = "Renal impairment"
    - "Renal disorder" -> ae_canonical = "Renal impairment"
    - "Hyperglycemic episode" -> ae_canonical = "Hyperglycemia"
    - "Hyperglycemia > 54 mg/dL" -> ae_canonical = "Hyperglycemia"

- ae_family:
  - A slightly broader grouping that clusters very closely related variants and subtypes.
  - Use the same ae_family for labels that are essentially the same clinical topic and would often be
    grouped together in a safety analysis (e.g., different subtypes of cholecystitis, or different
    phrasings of the same AE).
  - Examples:
    - If clinically reasonable, "Cholelithiasis" and "Cholecystitis" may share ae_family = "Gallbladder disease".
    - "Nausea" and "Vomiting" should NOT be merged into the same ae_family; each stays in its own family.
  - If no broader grouping is natural, set ae_family equal to ae_canonical.

- Summary rows:
  - Some labels are global summaries such as "Any adverse event", "Serious adverse events",
    "Adverse events leading to discontinuation".
  - These should be marked as is_summary = true and given an appropriate summary_type.
  - For summary rows, ae_canonical and ae_family may reflect the summary (e.g. "Any adverse event",
    "Serious adverse events") rather than a specific PT.

For EACH label I give you, output a JSON object with:
- "label_raw": the original label string.
- "ae_canonical": a short, specific canonical name as defined above.
- "ae_family": the broader family name as defined above. If no broader grouping is natural,
  set ae_family equal to ae_canonical.
- "is_summary": true if this label describes a global summary row
  (e.g. "Any adverse event", "Serious adverse events", "Adverse events leading to discontinuation"),
  otherwise false.
- "summary_type": if is_summary is true, one of:
  - "any_ae"
  - "serious_ae"
  - "fatal_ae"
  - "ae_leading_to_discontinuation"
  - "other_summary"
  Otherwise null.
- "notes": (optional) short explanation of your reasoning if needed.
- "confidence": "high", "medium", or "low".

Return ONLY a JSON list (array) of these objects, with one object per label, and no extra commentary.

Here are the labels:
"""
    body = "\n".join(f"- {lab}" for lab in labels)
    return instructions + "\n" + body


def build_arm_prompt(arms_batch: List[dict]) -> str:
    """
    Build prompt for analyzing clinical trial arms.
    
    Args:
        arms_batch: list of dicts with keys: trial_id, arm_label, drug_name, phase, indication, arm_role
    """
    instructions = """
You are helping analyze clinical trial arms for GLP-1 and related drugs.

For each trial arm, decide if it represents a clinically relevant dose that would
likely be used in real-world practice (approved dose or something very close to it),
as opposed to:
- clear sub-therapeutic exploratory doses
- run-in/titration-only schedules
- placebo or non-active comparator arms.

Guidelines:
- Placebo arms are NEVER clinically relevant.
- Arms with obviously tiny doses (e.g. early phase 0.25 mg where the drug is normally used at 1–2+ mg)
  are usually NOT clinically relevant unless that is an approved maintenance dose.
- If multiple doses are approved (e.g. 1.0 mg and 2.4 mg), they can all be clinically relevant.
- If you are unsure, be conservative and set is_clinical_dose = true.
- If it is not approved yet, typically select the dose with the best balance of efficacy and safety that would be a fair comparison to other approved drugs

For EACH arm I give you, output a JSON object with:
- "trial_id"
- "arm_label"
- "drug_name"
- "dose_mg": numeric dose if you can parse it from the arm_label (e.g. 1.0, 2.4, 50).
  If unclear, use null.
- "route": short route if obvious from context (e.g. "SC", "oral", "IV"), else null.
- "frequency": short dosing frequency (e.g. "QD", "BID", "QW", "Q2W", "Q4W"), else null.
- "is_placebo": true if this is clearly a placebo arm.
- "is_clinical_dose": true if this arm is a clinically relevant, real-world dose as defined above.
- "include_in_plot": true if you think this arm should be included when plotting
  dose vs adverse events; typically true for active clinically relevant doses, false for placebo and
  non-relevant exploratory doses.
- "notes": optional, brief explanation if needed.
- "confidence": "high", "medium", or "low".

Return ONLY a JSON list (array) of these objects, no extra commentary.

Here are the trial arms:
"""
    lines = []
    for arm in arms_batch:
        # provide a little structured context
        context = {
            "trial_id": arm["trial_id"],
            "drug_name": arm["drug_name"],
            "arm_label": arm["arm_label"],
            "arm_role": arm.get("arm_role"),
            "phase": arm.get("phase"),
            "indication": arm.get("indication"),
        }
        lines.append(json.dumps(context))
    return instructions + "\n" + "\n".join(lines)


def build_indication_prompt(labels: List[str]) -> str:
    """Build prompt for harmonizing indication labels."""
    instructions = """
You are harmonizing clinical trial indication labels.

Goal:
- Map raw indication strings from trial publications into a single, normalized
  indication_group that can be used for cross-trial analysis.

Rules:
- If two raw labels describe essentially the same patient population, they
  should share the SAME indication_group string.
- Keep important distinctions that change the biology or safety context, but collapse distinctions that a reasonable biotech analyst wouldn't care about (obesity vs overweight are the same thing and would all be obesity):

  e.g., "Type 2 diabetes" vs "Obesity without diabetes" vs
  "Obesity with type 2 diabetes".
- Normalize wording and spelling but keep groups intuitive and human-readable:
  e.g. "Type 2 diabetes", "Obesity", "Obesity + T2D", "NAFLD/NASH".
- If a label is very vague (e.g. "Overweight or obese with comorbidities"),
  choose a concise but reasonable indication_group such as
  "Obesity with cardiometabolic risk".

For EACH label I give you, output a JSON object with:
- "indication_raw": the original label string.
- "indication_group": a concise, normalized indication group as defined above.
- "notes": (optional) short explanation if needed.
- "confidence": "high", "medium", or "low".

Return ONLY a JSON list (array) of these objects, with one object per input label,
and no extra commentary.

Here are the raw indication labels:
"""
    body = "\n".join(f"- {lab}" for lab in labels)
    return instructions + "\n" + body


def parse_json_response(response_content: str) -> List[dict]:
    """
    Parse JSON response from OpenAI API, handling various wrapper formats.
    
    Args:
        response_content: Raw JSON string from API
    
    Returns:
        List of dictionaries
    """
    result = json.loads(response_content)
    
    # handle possible response wrappers (check for common keys)
    if isinstance(result, dict):
        if "results" in result:
            result = result["results"]
        elif "result" in result:
            result = result["result"]
        # If result is still a dict but not a list, try to find a list value
        if not isinstance(result, list):
            for key, value in result.items():
                if isinstance(value, list):
                    result = value
                    break
    
    # Ensure result is a list
    if not isinstance(result, list):
        raise ValueError(f"Expected a list but got {type(result)}: {result}")
    
    return result


def process_ae_labels(df_ae: pd.DataFrame, client: OpenAI, model: str = "gpt-4o") -> pd.DataFrame:
    """Process AE labels through OpenAI to get canonical and family mappings."""
    ae_labels = (
        df_ae["AE_label_raw"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    
    if not ae_labels:
        print("No AE labels found to process.")
        return df_ae
    
    print(f"Processing {len(ae_labels)} unique AE labels...")
    prompt = build_prompt_for_labels(ae_labels)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    result = parse_json_response(response.choices[0].message.content)
    ae_mapping = pd.DataFrame(result)
    
    # Merge into df_ae
    if "label_raw" in ae_mapping.columns:
        df_ae = df_ae.merge(
            ae_mapping[["label_raw", "ae_canonical", "ae_family", "is_summary", "summary_type"]],
            left_on="AE_label_raw",
            right_on="label_raw",
            how="left",
        )
    
    return df_ae


def process_arms(df_ae: pd.DataFrame, df_trials: pd.DataFrame, client: OpenAI, model: str = "gpt-4o") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process arms through OpenAI to get clinical dose information."""
    # Create df_arms
    df_arms = (
        df_ae[["trial_id", "arm_label", "arm_role", "drug_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df_arms = df_arms.merge(
        df_trials[["trial_id", "phase", "indication"]],
        on="trial_id",
        how="left",
    )
    
    print(f"Processing {len(df_arms)} unique arms...")
    prompt = build_arm_prompt(df_arms.to_dict(orient="records"))
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    result = parse_json_response(response.choices[0].message.content)
    arm_mapping = pd.DataFrame(result)
    
    # Merge into df_ae
    if "trial_id" in arm_mapping.columns and "arm_label" in arm_mapping.columns:
        df_ae = df_ae.merge(
            arm_mapping[["trial_id", "arm_label", "is_clinical_dose", "include_in_plot", "dose_mg", "route", "frequency"]],
            on=["trial_id", "arm_label"],
            how="left",
        )
    
    return df_ae, df_arms


def process_indications(df_trials: pd.DataFrame, client: OpenAI, model: str = "gpt-4o") -> pd.DataFrame:
    """Process indications through OpenAI to get indication groups."""
    indications = (
        df_trials["indication"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    
    if not indications:
        print("No indications found to process.")
        return df_trials
    
    print(f"Processing {len(indications)} unique indications...")
    prompt = build_indication_prompt(indications)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    result = parse_json_response(response.choices[0].message.content)
    indication_mapping = pd.DataFrame(result)
    
    # Verify required columns
    required_cols = ["indication_raw", "indication_group"]
    missing_cols = [col for col in required_cols if col not in indication_mapping.columns]
    if missing_cols:
        print(f"Warning: indication_mapping is missing columns: {missing_cols}")
        print(f"Available columns: {list(indication_mapping.columns)}")
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    # Merge into df_trials
    df_trials = df_trials.merge(
        indication_mapping[["indication_raw", "indication_group"]],
        left_on="indication",
        right_on="indication_raw",
        how="left",
    )
    
    df_trials.drop(columns=["indication_raw"], inplace=True, errors="ignore")
    
    return df_trials


def load_trial_json_files(trial_paths: List[Union[str, Path]]) -> List[dict]:
    """
    Load trial JSON files from file paths.
    
    Args:
        trial_paths: List of file paths to JSON files
    
    Returns:
        List of trial objects (each as a list containing one dict)
    """
    trial_data = []
    for path in trial_paths:
        path = Path(path)
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue
        
        with open(path, "r") as f:
            trial_obj = json.load(f)
        # Wrap in list to match expected format: [trial_obj]
        trial_data.append([trial_obj])
    
    return trial_data


def process_trials(
    trial_data: List[dict],
    openai_api_key: str,
    output_dir: Union[str, Path] = ".",
    model: str = "gpt-4o",
    save_ae_labels: bool = False,
) -> dict:
    """
    Main function to process trials and generate CSV outputs.
    
    Args:
        trial_data: List of trial objects (each as [trial_obj])
        openai_api_key: OpenAI API key for processing
        output_dir: Directory to save CSV files
        model: OpenAI model to use (default: "gpt-4o")
        save_ae_labels: Whether to save ae_labels_raw.csv
    
    Returns:
        Dictionary with paths to generated CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # Step 1: Load and process trial data
    print("Loading trial data...")
    df_trials, df_baseline, df_ae = load_reducto_results(trial_data)
    
    # Step 2: Merge drug_name into df_ae from df_trials
    df_ae = df_ae.merge(
        df_trials[["trial_id", "drug_name"]],
        on="trial_id",
        how="left",
    )
    
    # Step 3: Process AE labels
    print("\nProcessing AE labels...")
    df_ae = process_ae_labels(df_ae, client, model)
    
    # Optionally save AE labels raw
    if save_ae_labels:
        ae_labels = (
            df_ae["AE_label_raw"]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        ae_labels.to_csv(output_dir / "ae_labels_raw.csv", index=False)
        print(f"Saved ae_labels_raw.csv")
    
    # Step 4: Process arms
    print("\nProcessing arms...")
    df_ae, df_arms = process_arms(df_ae, df_trials, client, model)
    
    # Step 5: Process indications
    print("\nProcessing indications...")
    df_trials = process_indications(df_trials, client, model)
    
    # Step 6: Save CSV files
    print("\nSaving CSV files...")
    output_files = {
        "trials": output_dir / "trials.csv",
        "ae": output_dir / "ae.csv",
        "baseline": output_dir / "baseline.csv",
        "arms": output_dir / "arms.csv",
        "trials_with_indication_groups": output_dir / "trials_with_indication_groups.csv",
    }
    
    df_trials.to_csv(output_files["trials"], index=False)
    df_ae.to_csv(output_files["ae"], index=False)
    df_baseline.to_csv(output_files["baseline"], index=False)
    df_arms.to_csv(output_files["arms"], index=False)
    df_trials.to_csv(output_files["trials_with_indication_groups"], index=False)
    
    print("\n✓ Successfully generated all CSV files:")
    for name, path in output_files.items():
        print(f"  - {path}")
    
    return output_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process trial JSON files and generate CSV outputs")
    parser.add_argument("trial_files", nargs="+", help="Paths to trial JSON files")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--output-dir", default=".", help="Output directory for CSV files")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--save-ae-labels", action="store_true", help="Save ae_labels_raw.csv")
    
    args = parser.parse_args()
    
    # Load trial files
    trial_data = load_trial_json_files(args.trial_files)
    
    if not trial_data:
        print("Error: No trial files could be loaded.")
        exit(1)
    
    # Process trials
    process_trials(
        trial_data=trial_data,
        openai_api_key=args.api_key,
        output_dir=args.output_dir,
        model=args.model,
        save_ae_labels=args.save_ae_labels,
    )

