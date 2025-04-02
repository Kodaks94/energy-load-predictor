#UCI Energy Efficiency Dataset
import pandas as pd
import wandb
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
import itertools

energy_efficiency = fetch_ucirepo(id=242)
x = energy_efficiency.data.features
y = energy_efficiency.data.targets

x = x.loc[:, ~x.columns.str.contains('^Unnamed')]
x = x.dropna()

y1 = y['Y1'] #cooling
y2 = y['Y2'] #heating




x.columns = [
    "Relative Compactness", "Surface Area", "Wall Area", "Roof Area",
    "Overall Height", "Orientation", "Glazing Area", "Glazing Area Distribution"
]

scaler = StandardScaler()
def FeatureTypeScaler(Cftn,Nftn, data, scale_subset=None):

    if scale_subset is None:
        scale_subset = Nftn
    scaled = pd.DataFrame(index=data.index)
    for col in Nftn:
        if col in scale_subset:
            scaled[col] = scaler.fit_transform(data[[col]]).ravel()
        else:
            scaled[col] = data[col]  # Leave as-is

    # Encode categorical
    encoded = pd.get_dummies(data[Cftn], dtype=int)

    return pd.concat([scaled, encoded], axis=1)

def run_experiment(scaled_features, x_raw, y, Cftn, Nftn):
    identifier = "Scaled: " + (", ".join(scaled_features) if scaled_features else "None")
    config = {feature: (feature in scaled_features) for feature in Nftn}

    wandb.init(
        project="energy-efficiency-scaling",
        name=identifier,
        tags=["scaling-combo", "xgboost"],
        config= config,
        group="subset-scaling-experiments"
    )
    x = FeatureTypeScaler(Cftn, Nftn, x_raw, scale_subset=scaled_features)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators = 100, learning_rate= 0.1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Log results to W&B
    wandb.log({
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    wandb.finish()
    return model


numeric_cols = [
    "Relative Compactness", "Surface Area", "Wall Area",
    "Roof Area", "Overall Height", "Glazing Area"
]
categorical_cols = ["Orientation", "Glazing Area Distribution"]

# Try different combinations
combinations = [
    ["Relative Compactness"],
    ["Surface Area", "Wall Area"],
    ["Roof Area", "Overall Height"],
    ["Relative Compactness", "Glazing Area"],
    numeric_cols  # scale all
]

all_combinations = []
#for r in range(1, len(numeric_cols) + 1):
#    all_combinations += list(itertools.combinations(numeric_cols, r))
model = run_experiment([], x.copy(), y, categorical_cols, numeric_cols)
for combo in all_combinations:
    run_experiment(list(combo), x.copy(), y, categorical_cols, numeric_cols)


your_home = {
    "Relative Compactness": 0.75,       #
    "Surface Area": 551.2,              #Surface Area≈Wall Area+Roof Area+Floor Area=151.2+200+200=551.2m^2
    "Wall Area": 151.2,                 #Wall Area≈Perimeter×Height=56×2.7=151.2 m^2
    "Roof Area": 200.0,                 #Roof = Area≈200m^2
    "Overall Height": 2.7,
    "Orientation": 4,                    # South
    "Glazing Area": 0.2,                 # Estimated from 6 mixed windows
    "Glazing Area Distribution": 3       # Balanced (assumed)
}

your_home_df = pd.DataFrame([your_home])

x_input = FeatureTypeScaler(
    Cftn=categorical_cols,
    Nftn=numeric_cols,
    data=your_home_df,
    scale_subset=[]  # Make sure this matches your training combo
)

prediction = model.predict(x_input)
print("Predicted Heating and Cooling Load:", prediction, "kWh/m²")

import pandas as pd
import matplotlib.pyplot as plt

# Base input for your house
base_input = {
    "Relative Compactness": 0.75,
    "Surface Area": 551.2,
    "Wall Area": 151.2,
    "Roof Area": 200.0,
    "Overall Height": 2.7,
    "Orientation": 4,  # South
    "Glazing Area Distribution": 3
}

# Glazing area values to test
glazing_values = [0.0, 0.1, 0.25, 0.4]

heating_preds = []
cooling_preds = []

for ga in glazing_values:
    input_data = base_input.copy()
    input_data["Glazing Area"] = ga
    df = pd.DataFrame([input_data])

    # Preprocess for both models (assumes same scaler + logic used)
    x_input = FeatureTypeScaler(categorical_cols, numeric_cols, df, scale_subset=[])

    # Predict both heating and cooling
    preds = model.predict(x_input)
    heating_preds.append(preds[0][0])  # Y1
    cooling_preds.append(preds[0][1])  # Y2

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(glazing_values, heating_preds, marker='o', label="Heating Load (Y1)")
plt.plot(glazing_values, cooling_preds, marker='x', label="Cooling Load (Y2)")
plt.xlabel("Glazing Area")
plt.ylabel("Energy Load (kWh/m²)")
plt.title("Effect of Glazing Area on Heating and Cooling Load")
plt.legend()
plt.grid(True)
plt.show()
