# 🏡 Energy Load Predictor

This project uses machine learning (XGBoost) to predict a building's **heating** and **cooling energy requirements** based on its architectural and design features. It's built using the **UCI Energy Efficiency dataset**, and also allows users to estimate energy loads for their **own homes**.

---

## 🔍 What It Does

- Trains a regression model to predict:
  - 🔥 **Heating Load (Y1)** in kWh/m²  
  - ❄️ **Cooling Load (Y2)** in kWh/m²
- Uses features such as:
  - Building shape (compactness, surface area, height)
  - Orientation (North, South, etc.)
  - Glazing (amount and placement of windows)
- Supports **custom input**: enter your house’s features to predict its energy demand
- Performs **experiments** on different feature scaling combinations using **Weights & Biases** for tracking

---

## 💡 Key Features

- 🔧 **XGBoost Regressor** for accurate, fast predictions
- 🧪 **All combinations of scaled features** tested for impact on performance
- 📊 Evaluation metrics: `MSE`, `RMSE`, `MAE`, and `R²`
- 📈 Real-time logging with [Weights & Biases (wandb.ai)](https://wandb.ai/)
- 🏠 **User-defined house prediction** — test how design changes affect energy efficiency

---

## 🧠 Technologies Used

- Python 3
- XGBoost
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Weights & Biases (wandb)

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Kodaks94/energy-load-predictor.git
cd energy-load-predictor

# Create a virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main notebook or script
```

---

## 📦 Example Prediction

```python
# Example features for a real house
{
  "Relative Compactness": 0.75,
  "Surface Area": 551.2,
  "Wall Area": 151.2,
  "Roof Area": 200.0,
  "Overall Height": 2.7,
  "Orientation": 4,
  "Glazing Area": 0.2,
  "Glazing Area Distribution": 3
}
# Prediction → Heating Load: ~10.3, Cooling Load: ~13.6 kWh/m²
```

---

## 🧪 Results Snapshot

| Metric | Value (Example Run) |
|--------|---------------------|
| MSE    | 0.63                |
| RMSE   | 0.73                |
| MAE    | 0.47                |
| R²     | 0.993               |

---

## 📁 Folder Structure

```
├── data/                   # Raw or processed datasets
├── notebooks/              # Jupyter Notebooks for exploration and training
├── models/                 # Saved model files
├── wandb/                  # W&B run logs (ignored in .gitignore)
├── app/ (optional)         # Future: FastAPI or Streamlit app
├── README.md               # This file
```

---

## 🙋‍♂️ Author

**Kodaks94**  
Feel free to fork, use, or contribute!
