
# Dragon Real Estates Price Prediction Model
#


### Project Overview:
#
The Dragon Real Estates Price Prediction Model is developed for Dragon Real Estates, a hypothetical company striving to revolutionize the real estate industry by automating property price analysis. Dragon Real Estates aims to leverage machine learning models to provide faster and more accurate property valuations, helping buyers, sellers, and agents make informed decisions based on comprehensive data insights.

The company seeks to automate the traditionally manual and error-prone process of property price prediction, ensuring more accurate forecasts by considering various factors such as crime rates, pollution levels, property taxes, and proximity to essential amenities. This project represents a crucial step toward achieving that vision by building a predictive model using advanced machine learning techniques.

#
### Dataset Source:
#
The dataset consists of 506 observations of real estate properties with the following attributes:

- CRIM:    Crime rate per capita by town.
- ZN:      Proportion of residential land zoned for large lots.
- INDUS:   Proportion of non-retail business acres per town.
- CHAS:    Proximity to the Charles River (1 if the property borders the river, 0 otherwise).
- NOX:     Nitric oxide concentration (air pollution).
- RM:      Average number of rooms per dwelling.
- AGE:     Proportion of owner-occupied units built before 1940.
- DIS:     Weighted distances to five Boston employment centers.
- RAD:     Index of accessibility to radial highways.
- TAX:     Property tax rate per $10,000.
- PTRATIO: Pupil-teacher ratio by town.
- B:       Proportion of Black population in the town.
- LSTAT:   Percentage of lower-status population.
- MEDV:    Median value of homes (target variable).

#
Dataset Summary:

- Total Records: 506 properties.
- Missing Values: The dataset has very few missing values, particularly in the RM column (3 missing values).
- Features: 13 independent features, with MEDV as the target variable for house prices.
    
#

### Project Workflow:
#
#### 1. Data Preprocessing

- Data Cleaning: Handling missing values in the RM(Average number of rooms) column. 

- Exploratory Data Analysis: Visualizing data distributions and relationships between variables using histograms, correlation heatmaps, and scatter plots.

- Feature Engineering: No new features were created in this project, but the existing features were analyzed in depth.

#
#### 2. Data Splitting

The dataset was split into training and testing sets using two methods:


- Custom Train-Test Split: Implemented a random shuffle split for a balanced distribution.
- Stratified Shuffle Split: Ensured the CHAS feature is evenly distributed across training and test sets to avoid bias.

#
#### 3. Model Development

Three machine learning models were used for prediction:

- Linear Regression: A baseline model that captures the linear relationships between features and the target variable.

- Decision Tree Regressor: A non-linear model that splits the data based on feature values to improve prediction accuracy. Advantages: Can model non-linear relationships.

- Random Forest Regressor: An ensemble method combining multiple decision trees to improve generalization and reduce overfitting. Performed better than individual decision trees due to its bagging technique.
#

#### 4. Model Evaluation

The following metrics were used to evaluate all models:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

#

#### 5. Model Scores and Statistics

 I) Linear Regression:
        
- Mean MSE: 25.64
- Standard Deviation: 4.15

 II) Decision Tree Regressor:
    
- Mean MSE: 28.12
- Standard Deviation: 3.85

III) Random Forest Regressor:

- Mean MSE: 18.45
- Standard Deviation: 2.91
        
#

### Statistical Report:

A comprehensive statistical report is attached in the repository. This report includes:

- Detailed results of the Linear Regression, Decision Tree Regressor, and Random Forest Regressor models.

- Metrics comparisons, including training and test set performances.

- Insights into model performance using mean and standard deviation for error metrics.

#
### Installation and Setup:

Prerequisites-

Before running the project, make sure to install the necessary packages:

    pip install numpy pandas matplotlib scikit-learn


Running the Project:

- Clone the repository:

    git clone https://github.com/yourusername/dragon-estates-price-prediction


- Open the Jupyter notebook:

    jupyter notebook Real_Estate_Model.ipynb

- Run the notebook cells in sequence to reproduce the results, including data exploration, model training, and evaluation.

#
### Results and Insights:

The Random Forest Regressor provided the most accurate predictions with a low RMSE and outperformed other models.


#### Statistical Report:

Summary of all models: Linear Regression, Decision Tree, and Random Forest. Performance comparison on training and test data. Mean and standard deviation for MSE metrics for each model.

#
### Future Work:

- Implement Hyperparameter Tuning: Use techniques like Grid Search or Randomized Search to optimize model parameters.

-  Add More Features: External features like neighborhood crime rates, proximity to public transport, and economic trends could improve model accuracy.

- Explore advanced models like XGBoost or Gradient Boosting to improve performance further.
#

### Repository Structure:

     ├── Housing_Data.csv            # Dataset
     ├── Real_Estate_Model.ipynb     # Jupyter notebook containing the model code
     ├── Statistical_Report.pdf      # Detailed statistical report of models
     └── README.md                   # Project overview and instructions

#
### Contributing

Contributions are welcome! If you want to improve the model or add new features, feel free to submit a pull request.
