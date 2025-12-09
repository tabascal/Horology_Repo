import pandas as pd
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df = pd.read_csv("rolex_scaper_clean.csv")

# Filtrado básico
df = df.dropna(subset=["model", "condition", "price"])


# ------------------------------------------------------------
# Fair value calculator (completamente robusto)
# ------------------------------------------------------------
def compute_fair_value(group):
    """
    Ajusta una regresión lineal dentro de cada grupo (modelo + condición)
    manejando NaNs y columnas problemáticas.
    """

    numeric_cols = group.select_dtypes(include="number").columns.tolist()

    # Quitamos target
    if "price" in numeric_cols:
        numeric_cols.remove("price")

    # Si no hay features → media del precio
    if len(numeric_cols) == 0:
        return group["price"].mean()

    X = group[numeric_cols].copy()
    y = group["price"].copy()

    # 1) Eliminar columnas numéricas completamente vacías
    X = X.dropna(axis=1, how="all")

    # 2) Si después no queda nada → media del precio
    if X.shape[1] == 0:
        return y.mean()

    # 3) Imputación robusta
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # 4) Eliminar filas donde TODAS las features están NaN (después de imputar no debería ocurrir)
    X = X.dropna()
    y = y.loc[X.index]

    # 5) Entrena el modelo
    model = LinearRegression()
    model.fit(X, y)

    # Punto central del grupo
    mean_point = X.mean().to_frame().T

    fair_value = float(model.predict(mean_point)[0])
    return fair_value


# ------------------------------------------------------------
# Compute fair value for each Model × Condition
# ------------------------------------------------------------
results = []

for (model, condition), group in df.groupby(["model", "condition"]):
    fair_val = compute_fair_value(group)
    results.append({
        "model": model,
        "condition": condition,
        "fair_value": round(fair_val, 2)
    })

# Convert to dataframe
result_df = pd.DataFrame(results)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
for _, row in result_df.iterrows():
    print(
        f"Model: {row['model']} | Condition: {row['condition']} | Fair Value: {row['fair_value']} €")
