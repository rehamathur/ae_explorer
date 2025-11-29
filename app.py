import streamlit as st
import pandas as pd
import altair as alt


st.set_page_config("AE Explorer", layout="wide")


@st.cache_data
def load_ae_data():
    """Load AE data and optionally merge arm metadata."""
    ae = pd.read_csv("ae.csv")

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


@st.cache_data
def load_trials():
    """Load trial-level metadata if present."""
    try:
        trials = pd.read_csv("trials_with_indication_groups.csv")
    except FileNotFoundError:
        return pd.DataFrame()
    return trials


def aggregate_ae(ae: pd.DataFrame, label_col: str, clinical_only: bool) -> pd.DataFrame:
    """
    Aggregate AE rows at (AE label, trial, arm) level.

    - Optionally keep only clinically relevant arms (is_clinical_dose & include_in_plot).
    - Group by [label, trial, arm, drug, role, phase, indication] and take the mean of
      percent_with_event and n_with_event.
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


def parse_mapping(text: str):
    """
    Parse lines of 'key, value' into a dict {key: float(value)}.
    Lines starting with # or empty are ignored.
    """
    mapping = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        key, val = parts
        try:
            mapping[key] = float(val)
        except ValueError:
            continue
    return mapping


# ----------------- Load & basic settings -----------------
ae = load_ae_data()
trials = load_trials()

if ae.empty:
    st.error("No AE data found. Make sure 'ae.csv' is in the same folder.")
    st.stop()

st.title("Adverse Event (AE) Explorer")

with st.sidebar:
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
        trial_data = plot_df_adjusted[trial_mask]
        
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

# ----------------- NEW: AE vs trial-level numeric metadata from trials.csv -----------------
if trials.empty:
    st.info("No 'trials.csv' found in the app folder. Add it to enable trial-level plots.")
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

    # Numeric columns (this covers arbitrary numeric vars like duration, arm size, etc.)
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
