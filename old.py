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

    return ae


def aggregate_ae(ae: pd.DataFrame, label_col: str, clinical_only: bool) -> pd.DataFrame:
    """
    Aggregate AE rows:

    - Drop summary rows (is_summary == True) if present.
    - Optionally keep only clinically relevant arms
      (is_clinical_dose & include_in_plot).
    - Group by [label, trial, arm] and take the mean of percent_with_event
      and n_with_event.

    This is where we enforce: "If there are multiple AEs for a trial under
    the same family AE label, just take the mean value."
    """
    df = ae.copy()

    if clinical_only:
        if "is_clinical_dose" in df.columns:
            df = df[df["is_clinical_dose"]]
        if "include_in_plot" in df.columns:
            df = df[df["include_in_plot"]]

    # 3) Need the chosen label column
    df = df.dropna(subset=[label_col])

    # 4) Grouping keys
    group_cols = [label_col, "trial_id", "arm_label", "drug_name", "arm_role"]
    for col in ["phase", "indication"]:
        if col in df.columns:
            group_cols.append(col)

    # 5) Aggregate
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            percent_mean=("percent_with_event", "mean"),
            n_mean=("n_with_event", "mean"),
            arm_total_n=("arm_total_n", "max"),
            n_rows=("AE_label_raw", "count"),  # how many raw rows contributed
        )
    )

    return agg


# ----------------- UI -----------------

ae = load_ae_data()
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
        help="If checked, only non-control arms are shown.",
    )
    y_metric = st.radio(
        "Y-axis variable:",
        ["Percent with event", "Number with event"],
    )

agg = aggregate_ae(ae, label_col=label_col, clinical_only=clinical_only)

if agg.empty:
    st.warning("No data after filtering. Check your settings.")
    st.stop()

# AE selector
labels_sorted = sorted(agg[label_col].unique())
default_label = "Nausea" if "Nausea" in labels_sorted else labels_sorted[0]

selected_label = st.selectbox(
    f"Select {label_mode.lower()} to plot:",
    options=labels_sorted,
    index=labels_sorted.index(default_label),
)

plot_df = agg[agg[label_col] == selected_label].copy()

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

st.subheader(selected_label)

# Optional: show underlying table
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
    for extra in ["phase", "indication"]:
        if extra in plot_df.columns:
            show_cols.insert(1, extra)

    st.dataframe(
        plot_df[show_cols].sort_values(["trial_id", "arm_label"]),
        use_container_width=True,
    )
# Build label: Drug name + Arm
plot_df["drug_arm"] = (
    plot_df["drug_name"].fillna("") + " — " + plot_df["arm_label"].fillna("")
)

chart = (
    alt.Chart(plot_df)
    .mark_bar()
    .encode(
        y=alt.Y(
            "drug_arm:N",
            sort="-x",
            axis=alt.Axis(
                labelFontSize=12,
                title=None,
                labelOverlap=False,
                labelLimit=0,      # <- no truncation, show full sentence
                labelPadding=6,
            ),
        ),
        x=alt.X(
            f"{y_col}:Q",
            title=y_title,
            axis=alt.Axis(labelFontSize=11, titleFontSize=13),
        ),
        color=alt.Color("arm_role:N", title="Arm role"),
        tooltip=[
            "trial_id",
            "drug_name",
            "arm_label",
            "arm_role",
            alt.Tooltip(y_col, title=y_title),
            alt.Tooltip("arm_total_n", title="Arm N"),
            alt.Tooltip("n_rows", title="# raw AE rows in group"),
        ],
    )
    .properties(
        height=800,
        width=1000,    # widen chart so labels have room on the left
    )
)

st.altair_chart(chart, use_container_width=True)


st.caption(
    "Note: When grouping by AE family, if a trial+arm has multiple canonical AEs "
    "within that family, the app takes the mean percent/n across those raw rows."
)
