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
    Aggregate AE rows at (AE label, trial, arm) level.

    - Optionally keep only clinically relevant arms (is_clinical_dose & include_in_plot).
    - Group by [label, trial, arm, drug, role, phase, indication] and take the mean
      of percent_with_event and n_with_event.
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
    for col in ["drug_name", "arm_role", "phase", "indication"]:
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

    y_metric = st.radio(
        "Y-axis variable:",
        ["Percent with event", "Number with event"],
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
    for extra in ["phase", "indication"]:
        if extra in plot_df.columns and extra not in show_cols:
            show_cols.insert(1, extra)

    st.dataframe(
        plot_df[show_cols].sort_values([y_col], ascending=False),
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
        color=alt.Color("arm_role:N", title="Arm role")
        if "arm_role" in plot_df.columns
        else alt.value("steelblue"),
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
        height=max(400, 20 * len(plot_df)),
        width=900,
    )
)

st.altair_chart(chart, use_container_width=True)

# ----------------- OPTIONAL: AE vs metadata -----------------

with st.expander("Optional: plot this AE vs metadata (half-life, trial duration, etc.)"):
    st.markdown(
        "Enter metadata as `name, value` pairs (one per line).  \n"
        "- For half-life, **name = `drug_name`** (exactly as in the table above).  \n"
        "- For trial duration, **name = `trial_id`**."
    )

    col1, col2 = st.columns(2)

    with col1:
        half_life_text = st.text_area(
            "Half-life per drug (days)",
            value="",
            placeholder="Semaglutide, 7\nExenatide (Byetta), 0.25",
            height=120,
        )
    with col2:
        duration_text = st.text_area(
            "Trial duration per trial (weeks)",
            value="",
            placeholder="NCT01272219, 52",
            height=120,
        )

    half_life_map = parse_mapping(half_life_text)
    duration_map = parse_mapping(duration_text)

    meta_df = plot_df.copy()
    if half_life_map and "drug_name" in meta_df.columns:
        meta_df["half_life_days"] = meta_df["drug_name"].map(half_life_map)
    if duration_map:
        meta_df["trial_duration_weeks"] = meta_df["trial_id"].map(duration_map)

    numeric_options = []
    if "half_life_days" in meta_df.columns:
        numeric_options.append("half_life_days")
    if "trial_duration_weeks" in meta_df.columns:
        numeric_options.append("trial_duration_weeks")
    if "arm_total_n" in meta_df.columns:
        numeric_options.append("arm_total_n")

    display_names = {
        "half_life_days": "Half-life (days, manual)",
        "trial_duration_weeks": "Trial duration (weeks, manual)",
        "arm_total_n": "Arm total N (from data)",
    }

    if numeric_options:
        x_var = st.selectbox(
            "Metadata for X-axis",
            options=numeric_options,
            format_func=lambda c: display_names.get(c, c),
        )

        scatter_df = meta_df.dropna(subset=[x_var])

        if scatter_df.empty:
            st.info("No rows have a value for that metadata column yet.")
        else:
            scatter = (
                alt.Chart(scatter_df)
                .mark_circle(size=80)
                .encode(
                    x=alt.X(
                        f"{x_var}:Q",
                        title=display_names.get(x_var, x_var),
                    ),
                    y=alt.Y(
                        f"{y_col}:Q",
                        title=y_title,
                    ),
                    color=alt.Color("drug_name:N", title="Drug")
                    if "drug_name" in scatter_df.columns
                    else alt.value("steelblue"),
                    shape="arm_role:N"
                    if "arm_role" in scatter_df.columns
                    else alt.value("circle"),
                    tooltip=[
                        "trial_id",
                        "drug_name",
                        "arm_label",
                        "arm_role",
                        alt.Tooltip(x_var, title=display_names.get(x_var, x_var)),
                        alt.Tooltip(y_col, title=y_title),
                    ],
                )
                .properties(height=400)
            )
            st.altair_chart(scatter, use_container_width=True)
    else:
        st.info("Add at least one metadata value above to enable this plot.")
