import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------


@st.cache_data
def load_data():
    df = pd.read_csv("rolex_scaper_clean.csv")
    df = df.dropna(subset=["model", "condition", "price"])
    return df


df = load_data()


# ------------------------------------------------------------
# Fair value calculator
# ------------------------------------------------------------
def compute_fair_value(group):
    numeric_cols = group.select_dtypes(include="number").columns.tolist()

    # Remove target variable
    if "price" in numeric_cols:
        numeric_cols.remove("price")

    # If no numerical features are available → return mean price
    if len(numeric_cols) == 0:
        return group["price"].mean()

    X = group[numeric_cols].copy()
    y = group["price"].copy()

    # Remove fully empty columns
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        return y.mean()

    # Imputation
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Align indices
    X = X.dropna()
    y = y.loc[X.index]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Central point prediction
    mean_point = X.mean().to_frame().T
    fair_value = float(model.predict(mean_point)[0])

    return fair_value


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
st.title("🟩 Rolex Fair Value Calculator")

st.write(
    """
    Selecciona un **modelo** y una **condición**, y la aplicación calculará el
    *fair value* usando tu modelo de regresión interna por grupo.
    """
)

# Model selector
model_list = sorted(df["model"].unique())
selected_model = st.selectbox("Selecciona el modelo", model_list)

# Condition selector (filtered by model)
condition_list = sorted(
    df[df["model"] == selected_model]["condition"].unique())
selected_condition = st.selectbox("Selecciona la condición", condition_list)

# Compute button
if st.button("Calcular Fair Value"):
    group = df[(df["model"] == selected_model) &
               (df["condition"] == selected_condition)]

    fair_value = compute_fair_value(group)
    fair_value = round(fair_value, 2)

    st.success(f"💶  Fair Value estimado: **{fair_value} €**")
