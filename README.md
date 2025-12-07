This repository contains two related Random Forest case studies: one for predicting customer purchase of a holiday package, and one for predicting used car prices, each with its own dataset and notebook.

***

## Project Overview

This project demonstrates supervised machine learning workflows using Random Forest models on two business problems:  
- Holiday package uptake prediction for a travel company (“Wellness Tourism” targeting) using customer interaction data.  
- Used car price prediction using detailed vehicle attributes from the Indian market.

Both case studies follow a typical ML lifecycle: exploratory data analysis (EDA), feature engineering, model training, and evaluation within Jupyter notebooks.

***

## Repository Structure

- `RandomForest_Casestudy.ipynb` – Notebook for holiday package prediction using `Travel.csv` (classification: `ProdTaken`).
- `RandomForest_Casestudy2.ipynb` – Notebook for used car price prediction using `cardekho_imputated.csv` (regression: `selling_price`).
- `Travel.csv` – Customer-level travel dataset with demographics, contact, and behavior features for package purchase modeling.
- `cardekho_imputated.csv` – Used car dataset with vehicle specs and selling prices for price prediction.

***

## Data Description

### Travel Dataset (`Travel.csv`)

Each row represents a customer/prospect for a travel package campaign.
Key fields (non-exhaustive):

- `CustomerID`: Unique customer identifier.
- `ProdTaken`: Target variable indicating whether a package was purchased (binary).
- `Age`, `CityTier`, `DurationOfPitch`, `NumberOfFollowups`, `NumberOfTrips`, `MonthlyIncome`: Numeric engagement and demographic features.
- `TypeofContact`, `Occupation`, `Gender`, `ProductPitched`, `PreferredPropertyStar`, `MaritalStatus`, `Passport`, `OwnCar`, `Designation`: Categorical descriptors of profile and interaction.
- Derived features such as `NumberOfTotalVisitors` may be created in the notebook (e.g., aggregating visit-related fields).

### CarDekho Dataset (`cardekho_imputated.csv`)

Each row represents a used car listing with imputed values for missing data.
Key fields (non-exhaustive):

- `car_name`, `brand`, `model`: Text identifiers for the vehicle.
- `vehicle_age`: Age of the vehicle in years.
- `km_driven`: Total kilometers driven.
- `seller_type`: Seller category (e.g., individual, dealer).
- `fuel_type`, `transmission_type`: Powertrain and gearbox details.
- `mileage`, `engine`, `max_power`, `seats`: Technical specs and capacity.
- `selling_price`: Target variable (numeric used car price). Correlation analysis is performed between numeric features and `selling_price` in the notebook.

***

## How to Run

1. **Environment setup**  
   Ensure a Python environment with common data science libraries:  
   - `pandas`, `numpy`, `matplotlib`, `seaborn`, and typical ML stack (e.g., `scikit-learn`), as used in the notebooks.

2. **Open notebooks**  
   - Launch Jupyter Lab/Notebook.  
   - Open `RandomForest_Casestudy.ipynb` for the travel package classification case.
   - Open `RandomForest_Casestudy2.ipynb` for the car price regression case.

3. **Ensure data availability**  
   Place `Travel.csv` and `cardekho_imputated.csv` in the same directory as the notebooks or adjust the `pd.read_csv(...)` paths in the first data-loading cells.

4. **Execute cells sequentially**  
   Run all cells from top to bottom in each notebook to reproduce EDA, feature engineering, model training, and evaluation outputs.

***

## Use Cases and Extensions

- **Travel case study**:  
  - Build a production-ready classifier to prioritize which customers to target for the new Wellness Tourism package based on predicted `ProdTaken` probability.
  - Experiment with alternative models (e.g., gradient boosting), resampling strategies, or cost-sensitive learning.

- **Car price case study**:  
  - Deploy a regression model as a pricing API/tool for used car sellers to obtain suggested prices given vehicle attributes.
  - Extend feature engineering (e.g., brand/model embeddings, age buckets) and compare Random Forest with other regressors.

Adaptations can include hyperparameter tuning, cross-validation, feature importance analysis, and model interpretability methods (e.g., SHAP) on top of the provided baseline workflows.

[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56641498/5ccb2414-cc31-47f6-9fb8-c252aad6adc4/RandomForest_Casestudy.ipynb)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56641498/f8b42a9a-c2a8-4841-ba3f-2af1d1400be9/RandomForest_Casestudy2.ipynb)
