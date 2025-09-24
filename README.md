# Retail_Trade_Bundesbank_Analysis

This repository contains the files and software to run the Retail Trade Analysis on data from the Deutsche Bundesbank 


# Workflow

Step 1: Load data

src/data_loader.py fetches sector-level retail indices from the Bundesbank API.

Step 2: Feature engineering

src/features.py creates lagged variables, rolling statistics, growth rates, and target labels (6-month forward YoY).

Step 3: Model training

src/model.py trains a Logistic Regression classifier (wrapped in a pipeline with scaling + one-hot encoding).

Models are saved in serialized form (.pkl) for reuse in the Streamlit app.

Step 4: Inventory logic

src/inventory.py adjusts predictions based on inventory coverage, thresholds, and shortage amplification.

Step 5: Plots and Visualization

src/visualization.py produces:

a. Revenue/index forecast plots

b. Inventory coverage plots


Step 6: Pipeline orchestration

src/pipeline.py integrates all modules into a single workflow.

Run via main.py.

Step 7: Interactive dashboard

app/streamlit_app.py provides a Streamlit dashboard where users can:

a. Select sectors

b. Calibrate inventory and thresholds

c. View forecasts and plan recommendations

# How to Run

Step 1: Set up the dependencies listed in Requirements.txt. Install them with 

```
pip install -r Requirements.txt
```
Step 2: 

a. Run the pipeline (static analysis + plots)

```
python main.py
```
b. Run the Streamlit App (interactive dashboard)

```
streamlit run app/streamlit_app.py
```
