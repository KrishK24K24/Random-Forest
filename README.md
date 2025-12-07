# Random Forest Case Studies – Travel Package and Car Price Prediction

This repository contains two end-to-end machine learning case studies implemented in Jupyter notebooks: a classification model to predict holiday package purchase for a travel company, and a regression model to predict used car prices based on vehicle attributes.

Both case studies follow a typical data science lifecycle: exploratory data analysis, feature engineering, model building with Random Forests, and performance evaluation, using the datasets included in this repository.

***

## Project Structure

- RandomForest_Casestudy.ipynb  
  Jupyter notebook for the holiday package purchase prediction problem for a trips and travel company.

- RandomForest_Casestudy2.ipynb  
  Jupyter notebook for the used car price prediction problem using a CarDekho-style dataset.

- Travel.csv  
  Customer-level dataset used in RandomForest_Casestudy.ipynb for the travel package classification task.

- cardekho_imputated.csv  
  Used car dataset used in RandomForest_Casestudy2.ipynb for car price regression, with missing values already imputed.

The notebooks load their respective CSVs using direct calls to read the local files (for example, reading Travel.csv in the travel case and cardekho_imputated.csv in the car case).

***

## Problem Descriptions

### 1. Holiday Package Purchase Prediction (Travel Case Study)

A travel company wants to increase the effectiveness of its marketing for holiday packages, including a new wellness-oriented offering, by targeting customers who are most likely to purchase. Historical data suggests that only a minority of contacted customers actually buy a package, and previous campaigns contacted customers almost at random, leading to high marketing costs.

The goal of this case study is to build a classification model that predicts whether a customer will purchase a package, enabling focused outreach and more efficient marketing spend.

Key characteristics:

- Target variable  
  - ProdTaken: indicates whether the customer purchased a holiday package (binary classification).

- Input features (examples, not exhaustive)  
  - Customer and demographic information: CustomerID, Age, CityTier, MonthlyIncome.
  - Contact and engagement: TypeofContact (such as self enquiry or company invited), DurationOfPitch, NumberOfFollowups, NumberOfTrips.
  - Profile and lifestyle: Occupation, Gender, MaritalStatus, Passport, OwnCar, Designation.
  - Visit-related attributes: NumberOfPersonVisiting, NumberOfChildrenVisiting, and derived features such as NumberOfTotalVisitors created in the notebook.

The notebook performs exploratory analysis of these variables, including descriptive statistics for numeric fields like Age, DurationOfPitch, NumberOfFollowups, PreferredPropertyStar, NumberOfTrips, NumberOfChildrenVisiting, and MonthlyIncome.

### 2. Used Car Price Prediction (Car Price Case Study)

A company with data on used cars and their sale prices wants to estimate the price of a car based on its attributes, helping sellers receive realistic price suggestions aligned with current market conditions. This case study builds a regression model that predicts the numeric selling_price for each car record.

Key characteristics:

- Target variable  
  - selling_price: final transaction price of a used car (continuous numeric variable).

- Input features (examples, not exhaustive)  
  - Identification and brand: car_name, brand, model.
  - Usage and age: vehicle_age (in years), km_driven.
  - Seller and powertrain: seller_type (for example, individual or dealer), fuel_type, transmission_type.
  - Technical specifications: mileage, engine (displacement), max_power, seats.

The notebook includes tabular views of the dataset and correlation analysis among numerical variables such as vehicle_age, km_driven, mileage, engine, max_power, seats, and selling_price, revealing how engine size and max_power relate positively to price while higher vehicle_age and higher mileage tend to relate negatively.

***

## Data Description

### Travel.csv

Each row in Travel.csv represents a customer or prospect in a previous marketing campaign. The dataset is read directly into a pandas DataFrame in the travel notebook for preprocessing, analysis, and modeling.

Important columns include:

- CustomerID: Unique identifier for each customer.
- ProdTaken: Target indicator of package purchase.
- Age: Customer age in years (with some missing values that are handled during preprocessing).
- TypeofContact: How the lead was acquired or contacted, such as self-initiated or company-initiated.
- CityTier: Tier of city where the customer resides.
- DurationOfPitch: Duration of the sales pitch in minutes.
- Occupation: Employment type (for example, salaried, small business, freelancer).
- Gender: Customer gender.
- NumberOfPersonVisiting: Number of people travelling or visiting.
- NumberOfFollowups: How many follow-up contacts were made.
- ProductPitched: Type of package proposed (for example, Basic, Standard, Deluxe, Super Deluxe, King).
- PreferredPropertyStar: Preferred star rating of the property.
- MaritalStatus: Marital status category.
- NumberOfTrips: Number of trips taken by the customer.
- Passport: Indicator of whether the customer holds a passport.
- PitchSatisfactionScore: Subjective rating of satisfaction with the sales pitch.
- OwnCar: Whether the customer owns a car.
- NumberOfChildrenVisiting: Number of children included in the visit group.
- Designation: Customer designation level (for example, Executive, Manager, Senior Manager, AVP, VP).
- MonthlyIncome: Customer monthly income.

The notebook computes descriptive statistics for key numeric fields to understand distributions, ranges, and potential outliers before modeling.

### cardekho_imputated.csv

Each row in cardekho_imputated.csv represents a used car sold in the Indian market, with missing values already imputed so that the dataset can be used directly for training and evaluation. The notebook reads this dataset into a DataFrame as the starting point for car price modeling.

Important columns include:

- car_name: Combined name of the car (brand and model string).
- brand: Brand or manufacturer of the vehicle.
- model: Model name.
- vehicle_age: Age of the car in years at the time of sale.
- km_driven: Total kilometers driven.
- seller_type: Whether the seller is an individual or a dealer.
- fuel_type: Fuel category such as petrol or diesel.
- transmission_type: Gearbox type such as manual or automatic.
- mileage: Claimed mileage of the car.
- engine: Engine displacement in cubic centimeters.
- max_power: Peak engine power output.
- seats: Seating capacity.
- selling_price: Actual selling price, used as the target.

Correlation tables in the notebook show how numerical features are related to each other and to selling_price, which helps in feature selection and interpretation of the final model.

***

## Modeling Workflow

Both notebooks follow a broadly similar workflow, adjusted for classification and regression settings respectively.

Typical steps covered:

1. Data loading  
   - Read the CSVs into pandas DataFrames within the notebooks using straightforward file paths (Travel.csv for the travel study and cardekho_imputated.csv for the car study).

2. Exploratory Data Analysis (EDA)  
   - View head and descriptive statistics of the datasets.  
   - Inspect distributions of numerical variables such as Age, DurationOfPitch, mileage, engine displacement, and selling_price.
   - Examine value counts of categorical variables such as TypeofContact, ProductPitched, seller_type, and fuel_type.
   - For the car dataset, compute and inspect a correlation matrix to understand relationships like the strong positive correlation between engine and max_power and between these features and selling_price.

3. Data Cleaning and Feature Engineering  
   - Handle missing values in numerical and categorical columns, leveraging the already-imputed car dataset and applying imputations or derived fields on the travel dataset.
   - Create new features where appropriate, such as the total number of visitors by combining adults and children in the travel dataset.
   - Encode categorical variables into a numeric format suitable for Random Forest models (for example, via one-hot encoding or label encoding, depending on implementation in the notebooks).

4. Model Building  
   - Use Random Forest classifiers for the travel package case with ProdTaken as the target.
   - Use Random Forest regressors for the car price case with selling_price as the target.
   - Fit models on training data after splitting into training and validation sets.

5. Evaluation  
   - Compute evaluation metrics for the classification problem, such as accuracy and other classification metrics where implemented.
   - For the regression problem, use metrics such as error scores or R-squared to assess how well prices are predicted.
   - Inspect feature importance outputs from the Random Forest models to understand which variables drive predictions, such as income and engagement for package purchase or engine, power, and brand for car prices.

***

## How to Run and Extend

### Running the Notebooks

1. Place the following files in the same directory:  
   - RandomForest_Casestudy.ipynb  
   - RandomForest_Casestudy2.ipynb  
   - Travel.csv  
   - cardekho_imputated.csv

2. Set up a Python environment with at least:  
   - pandas  
   - numpy  
   - matplotlib  
   - seaborn  
   - a machine learning library such as scikit-learn for Random Forest models.

3. Launch Jupyter Notebook or JupyterLab in that directory.  

4. Open RandomForest_Casestudy.ipynb to work on the travel package prediction problem and run the cells from top to bottom.

5. Open RandomForest_Casestudy2.ipynb to work on the car price prediction problem and run the cells from top to bottom.

If your CSV files are in a different folder relative to the notebooks, update the file paths in the data-loading cells accordingly.

### Possible Extensions

- Perform systematic hyperparameter tuning for both Random Forest models to find better depth, number of trees, and splitting criteria.  
- Compare Random Forest with other algorithms such as gradient boosting, XGBoost, or linear models, especially for the car price regression.
- Implement cross-validation and cost-sensitive evaluation for the travel classification task to better reflect business objectives like minimizing wasted calls.
- Add more advanced feature engineering, such as bucketing vehicle_age, aggregating brand-level statistics, or encoding designations and occupations in a more informative way.
- Explore model explainability tools (for example, permutation importance or SHAP) to better communicate why certain customers or cars receive specific predictions.

This structure makes the repository a solid baseline for demonstrating applied Random Forest modeling on both marketing and pricing problems using real-world style tabular datasets.
