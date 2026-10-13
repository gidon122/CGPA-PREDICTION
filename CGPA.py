import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="CGPA Predictor", layout="wide")

DEFAULT_CSV = "synthetic_cgpa_practice_data.csv"
TARGET = "Projected_CGPA"
SEED = 42

ORDINALS = {
    "Level": ["100L", "200L", "300L", "400L", "500L"],
    "Study_Hours_Per_Week": ["Less than 5 hours", "5-9 hours", "10-14 hours",
                             "15-19 hours", "20-29 hours", "30 hours or more"],
    "Class_Attendance": ["Less than 50%", "50-69%", "70-89%", "90-100%"],
    "Continuous_Assessment_Score": ["Below 40%", "40-49%", "50-59%", "60-69%",
                                    "70-79%", "80% or above"],
    "Past_Questions_Practice": ["Never", "Rarely", "Occasionally", "Weekly", "Almost every day"],
    "Study_Groups_Participation": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
    "Sleep_Hours_Per_Night": ["Less than 4 hours", "4-5 hours", "6-7 hours",
                              "8-9 hours", "10 hours or more"],
    "Extracurricular_Hours_Per_Week": ["0 hour", "1-5 hours", "6-10 hours",
                                       "11-15 hours", "More than 15 hours"],
    "Work_Hustle": ["No", "Yes, less than 5 hours per week", "Yes, 5-10 hours per week",
                    "Yes, 11-20 hours per week", "Yes, more than 20 hours per week"],
}

# Only features a student can realistically change. direction: +1 raise, -1 lower
ACTIONABLE = {
    "Study_Hours_Per_Week":           (+1, "Increase your weekly study hours"),
    "Class_Attendance":               (+1, "Attend more of your classes"),
    "Continuous_Assessment_Score":    (+1, "Aim for higher scores in assignments, quizzes and tests"),
    "Past_Questions_Practice":        (+1, "Practise past questions more often"),
    "Study_Groups_Participation":     (+1, "Join study groups / tutorials more regularly"),
    "Work_Hustle":                    (-1, "Reduce the time you spend on work/business"),
    "Extracurricular_Hours_Per_Week": (-1, "Cut back time on extracurricular activities"),
}
MAXLEVEL = {c: len(ORDINALS[c]) - 1 for c in ACTIONABLE}

CAT_COLS = ["Department"]


# --------------------------------------------------------------------------
# Data loading and cleaning
# --------------------------------------------------------------------------
@st.cache_data
def load_and_clean(source):
    df = pd.read_csv(source).drop_duplicates()

    for col, order in ORDINALS.items():
        df[col] = df[col].map({v: i for i, v in enumerate(order)})

    # 1 = off-campus
    df["Accommodation"] = df["Accommodation"].str.startswith("Off", na=False).astype(int)

    # Real Google Form data: CGPA arrives as text, so force numeric and drop impossible values
    for col in ["Current_CGPA", "Last_SGPA", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < 0) | (df[col] > 5), col] = np.nan
    return df.dropna(subset=[TARGET, "Current_CGPA"])


# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
def make_pipeline(model, scale, num_feats):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("model", model)])


@st.cache_resource(show_spinner="Training models (this only runs once per dataset)...")
def train_models(df):
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    num_feats = [c for c in X.columns if c not in CAT_COLS]

    models = {
        "Linear Regression": make_pipeline(LinearRegression(), True, num_feats),
        "Random Forest": make_pipeline(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED),
            False, num_feats),
    }

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for name, pipe in models.items():
        cvr = cross_validate(
            pipe, X, y, cv=cv,
            scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"))
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "Model": name,
            "CV MAE": -cvr["test_neg_mean_absolute_error"].mean(),
            "CV RMSE": -cvr["test_neg_root_mean_squared_error"].mean(),
            "CV R2": cvr["test_r2"].mean(),
            "Test MAE": mean_absolute_error(y_te, pred),
            "Test RMSE": mean_squared_error(y_te, pred) ** 0.5,
            "Test R2": r2_score(y_te, pred),
        })
    results = pd.DataFrame(rows).set_index("Model").round(3)

    # Linear regression coefficients (standardised)
    lr = models["Linear Regression"]
    feat_names = lr.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(lr.named_steps["model"].coef_, index=feat_names)
    coefs = coefs.sort_values(key=abs, ascending=False)

    # Random forest permutation importance
    rf = models["Random Forest"]
    perm = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=SEED, scoring="r2")
    perm_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    # Pick on CV error (not test set), then refit on all data
    best_name = results["CV RMSE"].idxmin()
    final_model = clone(models[best_name]).fit(X, y)

    return {
        "results": results,
        "coefs": coefs,
        "perm_imp": perm_imp,
        "best_name": best_name,
        "final_model": final_model,
        "X_columns": list(X.columns),
        "num_feats": num_feats,
    }


def to_frame(student, X_columns, num_feats):
    """One-row DataFrame with the right column order and numeric dtypes."""
    row = pd.DataFrame([dict(student)])[X_columns]
    return row.astype({c: float for c in num_feats})


def recommend(student, art, top_k=3):
    """student: Series with the same columns as X (already encoded)."""
    model, cols, nums = art["final_model"], art["X_columns"], art["num_feats"]
    base = float(model.predict(to_frame(student, cols, nums))[0])
    gains = []
    for feat, (direction, advice) in ACTIONABLE.items():
        new_val = student[feat] + direction
        if new_val < 0 or new_val > MAXLEVEL[feat]:
            continue  # already at the limit
        trial = student.copy()
        trial[feat] = new_val
        delta = float(model.predict(to_frame(trial, cols, nums))[0]) - base
        if delta > 0.005:
            gains.append((delta, advice, feat))
    gains.sort(reverse=True)
    return base, gains[:top_k]


def show_recommendation(student, art):
    base, gains = recommend(student, art)
    st.metric("Predicted CGPA", f"{min(base, 5.0):.2f}")
    if not gains:
        st.success("You're already on a strong track. Keep your current habits.")
    for delta, advice, feat in gains:
        st.markdown(
            f"- **{advice}**: one step up/down on `{feat}` could add about **{delta:+.2f}** to your CGPA"
        )
    st.caption(
        "These are model estimates based on patterns in the survey data, "
        "not guarantees, and they show association rather than proof of cause."
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
st.title("CGPA Prediction & Study Advice")



try:
    df = load_and_clean(DEFAULT_CSV)
except FileNotFoundError:
    st.error(f"Training data `{DEFAULT_CSV}` was not found next to app.py.")
    st.stop()
except KeyError as e:
    st.error(f"The CSV is missing an expected column: {e}")
    st.stop()

art = train_models(df)
X = df.drop(columns=[TARGET])

tab_predict, tab_eda, tab_models, tab_imp = st.tabs(
    ["Predict & advise", "Data & EDA", "Model comparison", "Feature importance"]
)


# ---- Data & EDA ----
with tab_eda:
    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    st.subheader("Missing values")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values after cleaning.")
    else:
        st.dataframe(missing.to_frame("Missing"))

    st.subheader("Summary statistics")
    st.dataframe(df.describe().round(2).T, width="stretch")

    num_cols = [c for c in df.columns if c not in CAT_COLS]
    corr = df[num_cols].corr()

    st.subheader("Correlation with target")
    st.dataframe(
        corr[TARGET].drop(TARGET).sort_values(ascending=False).round(2).to_frame("Correlation"),
        width="stretch",
    )

    st.subheader("Distributions")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df[TARGET].hist(bins=20, ax=axes[0])
    axes[0].set_title("Projected CGPA distribution")
    df.boxplot(column=["Current_CGPA", "Last_SGPA", TARGET], ax=axes[1])
    axes[1].set_title("Outlier check")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Outliers (IQR rule)")
    out_rows = []
    for c in ["Current_CGPA", "Last_SGPA", "Current_Credit_Load", "Carryovers"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        out_rows.append({"Column": c, "Outliers": n_out})
    st.dataframe(pd.DataFrame(out_rows).set_index("Column"))

# ---- Model comparison ----
with tab_models:
    st.subheader("Model comparison")
    st.dataframe(art["results"], width="stretch")
    st.info(f"Best model on cross-validated RMSE: **{art['best_name']}** "
            "(refit on all the data and used for predictions).")
    st.download_button(
        "Download results (CSV)",
        art["results"].to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

# ---- Feature importance ----
with tab_imp:
    left, right = st.columns(2)
    with left:
        st.subheader("Linear Regression coefficients (standardised)")
        st.dataframe(art["coefs"].head(10).round(3).to_frame("Coefficient"),
                     width="stretch")
    with right:
        st.subheader("Random Forest permutation importance")
        fig, ax = plt.subplots(figsize=(7, 6))
        art["perm_imp"].sort_values().plot.barh(ax=ax, title="Permutation importance (RF)")
        fig.tight_layout()
        st.pyplot(fig)

# ---- Predict & advise ----
with tab_predict:
    mode = st.radio("Choose input", ["Enter my details", "Pick a student from the dataset"],
                    horizontal=True)

    if mode == "Pick a student from the dataset":
        idx = st.selectbox("Student row", X.index)
        st.dataframe(X.loc[[idx]], width="stretch")
        if st.button("Get advice"):
            show_recommendation(X.loc[idx], art)
    else:
        with st.form("student_form"):
            values = {}
            cols = st.columns(3)
            i = 0

            for c in X.columns:
                box = cols[i % 3]
                label = c.replace("_", " ")

                if c in CAT_COLS:
                    options = sorted(df[c].dropna().unique())
                    values[c] = box.selectbox(label, options)
                elif c in ORDINALS:
                    order = ORDINALS[c]
                    med = df[c].median()
                    default = int(round(med)) if pd.notna(med) else 0
                    choice = box.selectbox(label, order, index=default)
                    values[c] = order.index(choice)
                elif c == "Accommodation":
                    choice = box.selectbox(label, ["On-campus", "Off-campus"])
                    values[c] = 1 if choice == "Off-campus" else 0
                else:
                    col_min, col_max, col_med = df[c].min(), df[c].max(), df[c].median()
                    if (df[c].dropna() % 1 == 0).all():
                        values[c] = box.number_input(
                            label, min_value=int(col_min), max_value=int(col_max),
                            value=int(col_med), step=1)
                    else:
                        values[c] = box.number_input(
                            label, min_value=float(col_min), max_value=float(col_max),
                            value=float(col_med), step=0.01, format="%.2f")
                i += 1

            submitted = st.form_submit_button("Predict my CGPA")

        if submitted:
            show_recommendation(pd.Series(values)[X.columns], art)