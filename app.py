import streamlit as st
import pandas as pd
import altair as alt
import os
import json
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from reducto import AsyncReducto
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import from process_trials
from process_trials import (
    load_reducto_results,
    process_trials,
    load_trial_json_files,
)

st.set_page_config("AE Explorer", layout="wide")

# Pipeline ID for Reducto
REDUCTO_PIPELINE_ID = "k97e5ryq6b73xmqjbjxaedy8rn7vw541"

# ----------------- Helper Functions for PDF Processing -----------------

async def run_reducto(file_path: Path, client: AsyncReducto, log_messages: list):
    """Process a single PDF file through Reducto."""
    start = datetime.now()
    filename = file_path.name
    
    log_messages.append(f"⏳ Starting upload: {filename}")
    
    try:
        upload = await client.upload(file=file_path)
        log_messages.append(f"✅ Uploaded: {filename}")
        
        log_messages.append(f"🔄 Processing: {filename} through pipeline...")
        result = await client.pipeline.run(
            input=upload,
            pipeline_id=REDUCTO_PIPELINE_ID
        )
        
        elapsed = (datetime.now() - start).total_seconds()
        log_messages.append(f"✅ Completed: {filename} ({elapsed:.1f}s)")
        
        final_result = result.result.extract.result
        return final_result
    except Exception as e:
        log_messages.append(f"❌ Error processing {filename}: {str(e)}")
        raise


async def process_pdfs_with_reducto(pdf_files: list, log_messages: list):
    """Process multiple PDF files through Reducto asynchronously."""
    # Initialize Reducto client
    reducto_api_key = os.getenv("REDUCTO_API_KEY")
    if not reducto_api_key:
        raise ValueError("REDUCTO_API_KEY environment variable is not set. Please create a .env file.")
    
    client = AsyncReducto(api_key=reducto_api_key)
    
    # Process all PDFs in parallel
    tasks = [run_reducto(Path(file_path), client, log_messages) for file_path in pdf_files]
    results = await asyncio.gather(*tasks)
    
    return results


def save_trial_results(results: list, temp_dir: Path):
    """Save Reducto results as JSON files in the expected format."""
    trial_data = []
    
    for idx, result in enumerate(results):
        # result is already [{trial_obj}] format from Reducto, so append directly
        trial_data.append(result)
        
        # Save JSON file with unique name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"trial_{timestamp}_{idx}.json"
        json_path = temp_dir / filename
        with open(json_path, "w") as f:
            json.dump(result, f)
    
    return trial_data


# ----------------- Data Loading Functions -----------------

@st.cache_data
def load_ae_data(_ae_mtime, _arms_mtime, _trials_mtime):
    """
    Load AE data and optionally merge arm metadata.
    Uses file modification times to invalidate cache when files change.
    """
    try:
        ae = pd.read_csv("ae.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    # Try to pull in phase / indication from arms.csv if present
    try:
        arms = pd.read_csv("arms.csv")
        meta_cols = [c for c in ["trial_id", "arm_label", "phase", "indication"] if c in arms.columns]
        if meta_cols:
            meta = arms[meta_cols].drop_duplicates()
            ae = ae.merge(meta, on=["trial_id", "arm_label"], how="left")
    except FileNotFoundError:
        pass

    # Merge indication_group from trials CSV
    try:
        trials = pd.read_csv("trials_with_indication_groups.csv")
        if "indication_group" in trials.columns:
            trial_indication = trials[["trial_id", "indication_group"]].drop_duplicates()
            ae = ae.merge(trial_indication, on="trial_id", how="left")
    except FileNotFoundError:
        pass

    return ae


def _get_file_mtime(filename):
    """Get file modification time for cache invalidation."""
    try:
        return Path(filename).stat().st_mtime
    except FileNotFoundError:
        return 0


@st.cache_data
def load_trials(_trials_mtime):
    """
    Load trial-level metadata if present.
    Uses file modification time to invalidate cache when file changes.
    """
    try:
        trials = pd.read_csv("trials_with_indication_groups.csv")
    except FileNotFoundError:
        return pd.DataFrame()
    return trials


def aggregate_ae(ae: pd.DataFrame, label_col: str, clinical_only: bool) -> pd.DataFrame:
    """
    Aggregate AE rows at (AE label, trial, arm) level.
    """
    df = ae.copy()

    # Filter to clinically relevant arms if requested
    if clinical_only:
        if "is_clinical_dose" in df.columns:
            df = df[df["is_clinical_dose"]]
        if "include_in_plot" in df.columns:
            df = df[df["include_in_plot"]]

    # Require label
    df = df.dropna(subset=[label_col])

    # Grouping keys
    group_cols = [label_col, "trial_id", "arm_label"]
    for col in ["drug_name", "arm_role", "phase", "indication_group", "indication"]:
        if col in df.columns:
            group_cols.append(col)

    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            percent_mean=("percent_with_event", "mean"),
            n_mean=("n_with_event", "mean"),
            arm_total_n=("arm_total_n", "max"),
            n_rows=("AE_label_raw", "count"),
        )
    )
    return agg


# ----------------- Main App -----------------

st.title("Adverse Event (AE) Explorer")

# Create tabs for different functionality
tab1, tab2 = st.tabs(["📄 Process PDFs", "📊 Explore AEs"])

# ----------------- TAB 1: Process PDFs -----------------
with tab1:
    st.header("Upload and Process Trial PDFs")
    
    st.markdown("""
    Upload one or more trial PDF files to extract safety table data and generate CSV files.
    The PDFs will be processed through Reducto to extract trial metadata, baseline tables, and adverse event tables.
    """)
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    reducto_key = os.getenv("REDUCTO_API_KEY")
    
    if not openai_key or not reducto_key:
        st.warning("⚠️ **API Keys Required**")
        st.markdown("""
        Please create a `.env` file in the project directory with:
        ```
        OPENAI_API_KEY=your-openai-api-key
        REDUCTO_API_KEY=your-reducto-api-key
        ```
        """)
        st.stop()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to process"
    )
    
    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} file(s) selected for processing")
        
        # Check for existing CSV files
        existing_files = []
        for csv_file in ["ae.csv", "trials.csv", "arms.csv", "baseline.csv", "trials_with_indication_groups.csv"]:
            if Path(csv_file).exists():
                existing_files.append(csv_file)
        
        if existing_files:
            st.info(f"ℹ️ **Append mode:** New data will be appended to existing CSV files: {', '.join(existing_files)}")
        
        # Model selection
        model = st.selectbox(
            "OpenAI Model",
            options=["gpt-5"],
            index=0,
            help="Model to use for harmonizing labels"
        )
        
        # Process button
        if st.button("🚀 Process PDFs", type="primary"):
            log_messages = []
            log_container = st.empty()
            
            def update_log():
                """Update the log display"""
                log_container.text_area(
                    "Processing log",
                    "\n".join(log_messages[-30:]),
                    height=300,
                    disabled=True,
                    label_visibility="visible"
                )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                try:
                    # Save uploaded files temporarily
                    log_messages.append("📁 Saving uploaded PDFs...")
                    update_log()
                    pdf_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = temp_path / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        pdf_paths.append(str(file_path))
                        log_messages.append(f"  ✓ Saved: {uploaded_file.name}")
                        update_log()
                    
                    # Step 1: Process PDFs through Reducto
                    with st.status("🔄 Processing PDFs through Reducto...", expanded=True) as status:
                        log_messages.append(f"\n🔄 Processing {len(pdf_paths)} PDF file(s) through Reducto...")
                        status.write("📤 Uploading files and processing through pipeline...")
                        update_log()
                        
                        results = asyncio.run(
                            process_pdfs_with_reducto(pdf_paths, log_messages)
                        )
                        
                        # Show recent log messages
                        for msg in log_messages[-10:]:
                            if msg.strip():
                                status.write(msg)
                        
                        if not results:
                            status.error("❌ No results returned from Reducto processing.")
                            st.stop()
                        
                        status.update(label="✅ Reducto processing complete!", state="complete")
                        log_messages.append("\n✅ Reducto processing complete!")
                        update_log()
                    
                    # Step 2: Save results and prepare for CSV generation
                    with st.status("💾 Preparing trial data...", expanded=True) as status:
                        status.write("📋 Saving trial results...")
                        trial_data = save_trial_results(results, temp_path)
                        status.write(f"✓ Prepared {len(trial_data)} trial(s) for CSV generation")
                        status.update(label="✅ Trial data prepared!", state="complete")
                    
                    # Step 3: Generate CSVs using process_trials
                    with st.status("⚙️ Generating CSV files with AI harmonization...", expanded=True) as status:
                        status.write("🤖 This may take a few minutes...")
                        status.write("   - Processing AE labels through OpenAI")
                        status.write("   - Processing arms through OpenAI")
                        status.write("   - Processing indications through OpenAI")
                        
                        # Load existing CSV files if they exist
                        existing_data = {}
                        csv_files = {
                            "trials": "trials.csv",
                            "ae": "ae.csv",
                            "baseline": "baseline.csv",
                            "arms": "arms.csv",
                            "trials_with_indication_groups": "trials_with_indication_groups.csv",
                        }
                        
                        for key, filename in csv_files.items():
                            if Path(filename).exists():
                                try:
                                    existing_data[key] = pd.read_csv(filename)
                                    status.write(f"📂 Loaded existing {filename} ({len(existing_data[key])} rows)")
                                except Exception as e:
                                    status.write(f"⚠️ Could not load {filename}: {e}")
                                    existing_data[key] = pd.DataFrame()
                            else:
                                existing_data[key] = pd.DataFrame()
                        
                        # Redirect print statements to capture process_trials output
                        import sys
                        from io import StringIO
                        
                        old_stdout = sys.stdout
                        sys.stdout = StringIO()
                        
                        try:
                            # Process new trials to temp directory first
                            temp_output_dir = temp_path / "new_csvs"
                            temp_output_dir.mkdir(exist_ok=True)
                            
                            output_files = process_trials(
                                trial_data=trial_data,
                                openai_api_key=openai_key,
                                output_dir=temp_output_dir,
                                model=model,
                                save_ae_labels=False,  # Don't overwrite ae_labels_raw.csv
                            )
                            
                            # Capture output from process_trials and show in status
                            output = sys.stdout.getvalue()
                            if output:
                                for line in output.strip().split("\n"):
                                    if line.strip():
                                        status.write(line)
                                        log_messages.append(line)
                            
                            # Load new data
                            status.write("\n📊 Combining with existing data...")
                            new_data = {}
                            for key, path in output_files.items():
                                new_path = temp_output_dir / Path(path).name
                                if new_path.exists():
                                    new_data[key] = pd.read_csv(new_path)
                                    status.write(f"  ✓ Loaded new {key} data ({len(new_data[key])} rows)")
                            
                            # Append new data to existing data
                            combined_data = {}
                            for key in csv_files.keys():
                                if key in new_data:
                                    if not existing_data[key].empty:
                                        # Append new to existing
                                        combined_data[key] = pd.concat(
                                            [existing_data[key], new_data[key]],
                                            ignore_index=True
                                        )
                                        status.write(f"  ✓ Combined {key}: {len(existing_data[key])} existing + {len(new_data[key])} new = {len(combined_data[key])} total")
                                    else:
                                        # No existing data, just use new
                                        combined_data[key] = new_data[key]
                                        status.write(f"  ✓ Created new {key} ({len(combined_data[key])} rows)")
                                elif not existing_data[key].empty:
                                    # Keep existing if no new data
                                    combined_data[key] = existing_data[key]
                            
                            # Save combined data
                            status.write("\n💾 Saving combined CSV files...")
                            final_output_files = {}
                            for key, filename in csv_files.items():
                                if key in combined_data:
                                    final_path = Path(filename)
                                    combined_data[key].to_csv(final_path, index=False)
                                    final_output_files[key] = str(final_path)
                                    status.write(f"  ✓ Saved {filename} ({len(combined_data[key])} rows)")
                            
                            output_files = final_output_files
                            
                        finally:
                            sys.stdout = old_stdout
                        
                        status.update(label="✅ CSV files generated and appended successfully!", state="complete")
                    
                    # Success message
                    st.success("🎉 **Processing Complete!**")
                    
                    st.markdown("### Generated Files:")
                    for name, path in output_files.items():
                        if Path(path).exists():
                            file_size = Path(path).stat().st_size / 1024  # KB
                            st.markdown(f"- **{name}**: `{path}` ({file_size:.1f} KB)")
                    
                    # Show summary
                    try:
                        df_trials = pd.read_csv("trials.csv")
                        df_ae = pd.read_csv("ae.csv")
                        
                        st.markdown("### Summary:")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Trials", len(df_trials))
                        with col2:
                            st.metric("AE Records", len(df_ae))
                        with col3:
                            unique_aes = df_ae["AE_label_raw"].nunique() if "AE_label_raw" in df_ae.columns else 0
                            st.metric("Unique AEs", unique_aes)
                    except Exception as e:
                        st.warning(f"Could not load summary: {e}")
                    
                    # Clear cache so new data loads
                    load_ae_data.clear()
                    load_trials.clear()
                    
                    st.info("💡 Switch to the 'Explore AEs' tab to visualize the data!")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)

# ----------------- TAB 2: Explore AEs (Existing Functionality) -----------------
with tab2:
    # Get file modification times for cache invalidation
    ae_mtime = _get_file_mtime("ae.csv")
    arms_mtime = _get_file_mtime("arms.csv")
    trials_mtime = _get_file_mtime("trials_with_indication_groups.csv")
    
    ae = load_ae_data(ae_mtime, arms_mtime, trials_mtime)
    trials = load_trials(trials_mtime)

    if ae.empty:
        st.warning("⚠️ No AE data found. Please process PDF files first using the 'Process PDFs' tab.")
        st.info("""
        To get started:
        1. Go to the **Process PDFs** tab
        2. Upload one or more trial PDF files
        3. Click "Process PDFs" to generate the CSV files
        4. Return to this tab to explore the data
        """)
    else:
        st.header("Explore Adverse Events")

        # Sidebar with all settings
        with st.sidebar:
            # Cache clear button at top
            if st.button("🔄 Clear Cache & Reload", help="Clear Streamlit cache and reload data from CSV files"):
                load_ae_data.clear()
                load_trials.clear()
                st.rerun()
            
            st.markdown("---")
            st.header("Settings")

            label_mode = st.radio(
                "Group AEs by:",
                ["Canonical AE label", "AE family label"],
            )
            label_col = "ae_canonical" if "Canonical" in label_mode else "ae_family"

            clinical_only = st.checkbox(
                "Only clinically relevant arms",
                help="Filters to rows where is_clinical_dose & include_in_plot are True.",
                value=False,
            )

            hide_control = st.checkbox(
                "Hide control arms",
                value=False,
                help="If checked, only non-control arms are shown in plots.",
            )

            # Filter by indication_group (categorical)
            if "indication_group" in ae.columns:
                indication_groups = sorted(ae["indication_group"].dropna().unique())
                if indication_groups:
                    selected_indication_groups = st.multiselect(
                        "Filter by indication group",
                        options=indication_groups,
                        default=indication_groups,
                    )
                    if selected_indication_groups and len(selected_indication_groups) < len(indication_groups):
                        ae = ae[ae["indication_group"].isin(selected_indication_groups)]
            # Fallback to indication if indication_group not available
            elif "indication" in ae.columns:
                indications = sorted(ae["indication"].dropna().unique())
                if indications:
                    selected_indications = st.multiselect(
                        "Filter by indication",
                        options=indications,
                        default=indications,
                    )
                    if selected_indications and len(selected_indications) < len(indications):
                        ae = ae[ae["indication"].isin(selected_indications)]

            y_metric = st.radio(
                "Y-axis variable:",
                ["Percent with event", "Number with event"],
            )
            
            placebo_adjust = st.radio(
                "Placebo adjustment:",
                ["Raw values", "Placebo-adjusted"],
                help="Placebo-adjusted subtracts control arm values from treatment arms within each trial. Arms without controls remain unchanged.",
            )

        # Main content area
        agg = aggregate_ae(ae, label_col=label_col, clinical_only=clinical_only)
        if agg.empty:
            st.warning("No data after filtering. Check your settings.")
            st.stop()

        # ----------------- AE selection -----------------
        labels_sorted = sorted(agg[label_col].dropna().unique())
        default_label = "Nausea" if "Nausea" in labels_sorted else labels_sorted[0]

        selected_label = st.selectbox(
            f"Select {label_mode.lower()} to plot:",
            options=labels_sorted,
            index=labels_sorted.index(default_label) if default_label in labels_sorted else 0,
        )

        plot_df = agg[agg[label_col] == selected_label].copy()

        # Optionally drop control arms
        if hide_control and "arm_role" in plot_df.columns:
            plot_df = plot_df[plot_df["arm_role"] != "control"]

        if plot_df.empty:
            st.info("No rows for that AE with the current filters.")
            st.stop()

        # Decide which metric to show
        if "Percent" in y_metric:
            y_col = "percent_mean"
            y_title = "% with event (mean across raw rows)"
        else:
            y_col = "n_mean"
            y_title = "Patients with event (mean across raw rows)"

        # Apply placebo adjustment if requested
        if placebo_adjust == "Placebo-adjusted" and "arm_role" in plot_df.columns and "trial_id" in plot_df.columns:
            # Create a copy for adjustment
            plot_df_adjusted = plot_df.copy()
            plot_df_adjusted["has_placebo_adjustment"] = False
            
            # For each trial, find control values and adjust treatment arms
            for trial_id in plot_df_adjusted["trial_id"].unique():
                trial_mask = plot_df_adjusted["trial_id"] == trial_id
                
                # Find control arm(s) for this trial
                control_mask = trial_mask & (plot_df_adjusted["arm_role"] == "control")
                treatment_mask = trial_mask & (plot_df_adjusted["arm_role"] == "treatment")
                
                if control_mask.any() and treatment_mask.any():
                    # Get control value (take mean if multiple controls)
                    control_value = plot_df_adjusted.loc[control_mask, y_col].mean()
                    
                    # Adjust treatment arms
                    plot_df_adjusted.loc[treatment_mask, y_col] = (
                        plot_df_adjusted.loc[treatment_mask, y_col] - control_value
                    )
                    plot_df_adjusted.loc[treatment_mask, "has_placebo_adjustment"] = True
            
            plot_df = plot_df_adjusted
            
            # Remove control arms from plot when placebo adjusting
            if "arm_role" in plot_df.columns:
                plot_df = plot_df[plot_df["arm_role"] != "control"]
            
            if plot_df.empty:
                st.info("No treatment arms available after placebo adjustment.")
                st.stop()
            
            # Update y_title to reflect adjustment
            if "Percent" in y_metric:
                y_title = "% with event (placebo-adjusted)"
            else:
                y_title = "Patients with event (placebo-adjusted)"
        else:
            # Add column for consistency (all False when not adjusting)
            plot_df["has_placebo_adjustment"] = False

        st.subheader(selected_label)

        # ----------------- Underlying table -----------------
        with st.expander("Show underlying aggregated data"):
            show_cols = [
                "trial_id",
                "arm_label",
                "drug_name",
                "arm_role",
                "arm_total_n",
                "percent_mean",
                "n_mean",
                "n_rows",
            ]
            # Prioritize indication_group over indication for display
            for extra in ["phase", "indication_group", "indication"]:
                if extra in plot_df.columns and extra not in show_cols:
                    show_cols.insert(1, extra)

            # Sort by indication_group first (if available), then by the metric
            sort_cols = []
            if "indication_group" in plot_df.columns:
                sort_cols.append("indication_group")
            elif "indication" in plot_df.columns:
                sort_cols.append("indication")
            sort_cols.append(y_col)
            
            st.dataframe(
                plot_df[show_cols].sort_values(sort_cols, ascending=[True, False]),
                use_container_width=True,
            )

        # ----------------- Horizontal bar chart -----------------
        # Build label: Drug name + Arm
        if "drug_name" in plot_df.columns:
            plot_df["drug_arm"] = (
                plot_df["drug_name"].fillna("") + " — " + plot_df["arm_label"].fillna("")
            )
        else:
            plot_df["drug_arm"] = plot_df["arm_label"].fillna("")

        # Determine color encoding based on placebo adjustment
        if placebo_adjust == "Placebo-adjusted" and "has_placebo_adjustment" in plot_df.columns:
            # Use different colors for adjusted vs non-adjusted arms
            color_encoding = alt.Color(
                "has_placebo_adjustment:N",
                title="Placebo adjusted",
                scale=alt.Scale(
                    domain=[True, False],
                    range=["#2E86AB", "#A23B72"]  # Blue for adjusted, Purple for non-adjusted
                ),
                legend=alt.Legend(
                    title="Placebo adjusted",
                    values=[True, False],
                    labelExpr="datum.value == true ? 'Yes' : 'No'"
                )
            )
        elif "arm_role" in plot_df.columns:
            color_encoding = alt.Color("arm_role:N", title="Arm role")
        else:
            color_encoding = alt.value("steelblue")

        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "drug_arm:N",
                    sort="-x",
                    axis=alt.Axis(
                        title=None,
                        labelFontSize=12,
                        labelOverlap=False,
                        labelLimit=0,
                    ),
                ),
                x=alt.X(
                    f"{y_col}:Q",
                    title=y_title,
                    axis=alt.Axis(labelFontSize=11, titleFontSize=13),
                ),
                color=color_encoding,
                tooltip=[
                    "trial_id",
                    "drug_name",
                    "arm_label",
                    "arm_role",
                    alt.Tooltip(y_col, title=y_title, format=".2f"),
                    alt.Tooltip("arm_total_n", title="Arm N"),
                    alt.Tooltip("n_rows", title="# raw AE rows in group"),
                ] + ([alt.Tooltip("has_placebo_adjustment:N", title="Placebo adjusted")]
                    if placebo_adjust == "Placebo-adjusted" and "has_placebo_adjustment" in plot_df.columns
                    else []),
            )
            .properties(
                height=max(400, 20 * len(plot_df)),
                width=900,
            )
        )

        st.altair_chart(chart, use_container_width=True)

        # ----------------- Trial-level metadata scatter plot -----------------
        if trials.empty:
            st.info("No 'trials_with_indication_groups.csv' found. Process PDFs to generate trial metadata.")
        else:
            # Merge current AE rows with trial-level metadata
            merge_cols = [c for c in trials.columns if c != "trial_id" and c not in plot_df.columns]
            if merge_cols:
                trials_subset = trials[["trial_id"] + merge_cols]
            else:
                trials_subset = trials[["trial_id"]]

            trial_meta_df = plot_df.merge(trials_subset, on="trial_id", how="left")

            st.markdown("**Trial-level metadata (merged for this AE):**")
            st.dataframe(trial_meta_df, use_container_width=True)

            # Numeric columns
            numeric_cols = trial_meta_df.select_dtypes(include="number").columns.tolist()
            # Don't use y_col itself as X by default
            if y_col in numeric_cols:
                numeric_cols.remove(y_col)

            if not numeric_cols:
                st.info("No numeric columns available in merged AE + trial data for scatter.")
            else:
                x_var = st.selectbox(
                    "X-axis numeric variable (from AE/trial metadata)",
                    options=numeric_cols,
                    format_func=lambda c: c.replace("_", " "),
                )
                scatter_df2 = trial_meta_df.dropna(subset=[x_var])
                if scatter_df2.empty:
                    st.info(f"No rows have a value for '{x_var}' yet.")
                else:
                    scatter2 = (
                        alt.Chart(scatter_df2)
                        .mark_circle(size=80)
                        .encode(
                            x=alt.X(f"{x_var}:Q", title=x_var.replace("_", " ")),
                            y=alt.Y(f"{y_col}:Q", title=y_title),
                            color=alt.Color("drug_name:N", title="Drug")
                            if "drug_name" in scatter_df2.columns
                            else alt.value("steelblue"),
                            shape="arm_role:N"
                            if "arm_role" in scatter_df2.columns
                            else alt.value("circle"),
                            tooltip=[
                                "trial_id",
                                "drug_name",
                                "arm_label",
                                "arm_role",
                                alt.Tooltip(x_var, title=x_var.replace("_", " ")),
                                alt.Tooltip(y_col, title=y_title),
                            ],
                        )
                        .properties(height=400)
                    )
                    st.altair_chart(scatter2, use_container_width=True)
