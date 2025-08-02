import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from graphviz import Source

# ==== CONFIG ====
input_file = "in/enriched_candles.csv"
output_file = "out/DecisionTreeResult"

# Model hyperparameters
min_samples_leaf = 20
max_depth = 2
min_impurity_decrease = 0.001

# === Load dataframe ===
df = pd.read_csv(input_file, sep=",", encoding='cp1252')

# Filter data for signal to explain
df = df[df['IsBullish'] == True]
df = df[df['IsIncreasingVolume'] == True]
df = df[df['OR Breakout'] == True]
df = df[df['hour'].isin([7, 8, 9])]

# === Choose target column ===
y_column = "Next Bullish"

# === Choose features ===
# 0 - Volume
# 1 - RSI
# 2 - vix_close
# 10 - HasLongUpperWick
# 11 - HasLongLowerWick
# 12 - hour
# 29 - PrevDayHigh Breakout

x_column_indices = [0, 1, 10, 11, 12, 29]
x_columns = df.iloc[:, x_column_indices]

df = df.dropna(subset=[y_column] + list(x_columns.columns))
df_y = df[y_column]
df_X = df[x_columns.columns]

# === Identify feature types ===
categorical_cols = df_X.select_dtypes(include=['object', 'category']).columns.tolist()
boolean_cols = df_X.select_dtypes(include=['bool']).columns.tolist()
numerical_cols = df_X.select_dtypes(include=[np.number]).columns.tolist()

# Convert booleans to int
df_X.loc[:, boolean_cols] = df_X[boolean_cols].astype(int)

# === Set class weights ===
manual_class_weights = {
    0: 1.0,  # weight for class 0
    1: 1.0   # weight for class 1 (minority)
}

# === Build preprocessing pipeline ===
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', dtype=bool), categorical_cols),
        ('num', 'passthrough', numerical_cols + boolean_cols)
    ],
    remainder='drop'
)

# === Build model pipeline with manual class weights ===
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        class_weight=manual_class_weights,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        min_impurity_decrease=min_impurity_decrease
    ))
])

# === Fit model ===
clf.fit(df_X, df_y)

# === Evaluate ===
predictions = clf.predict(df_X)
print('Accuracy:', accuracy_score(df_y, predictions))
print('Balanced Accuracy:', balanced_accuracy_score(df_y, predictions))

# === Get feature names post preprocessing ===
feature_names = clf.named_steps['preprocessor'].get_feature_names_out()

# === Export tree with default visualization ===
dot_data = export_graphviz(
    clf.named_steps['classifier'],
    out_file=None,
    feature_names=feature_names,
    filled=True,       # color nodes by class
    rounded=True,
    special_characters=True,
    proportion=False,  # show absolute counts, not proportions
    label='all'        # show class name, gini, samples, value
)

graph = Source(dot_data)
graph.render(output_file)

print(f"Decision tree exported as '{output_file}.pdf' (or .png if you change format)")


