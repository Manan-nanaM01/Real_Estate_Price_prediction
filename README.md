# Real_Estate_Price_prediction
Real Estate Price Prediction
This project involves building a machine learning model to predict real estate prices based on various features like area, number of bedrooms, bathrooms, parking availability, and more. The dataset used in this project includes attributes related to the features of houses and their corresponding prices.

Technologies Used:
Python: Programming language used for data processing, visualization, and model building.

Pandas: Data manipulation and analysis library.

Matplotlib & Seaborn: Data visualization libraries for plotting graphs and distribution plots.

Scikit-learn: Machine learning library for model training, evaluation, and splitting datasets.

LabelEncoder: For encoding categorical variables into numerical values.

Dataset:
The dataset used for this project is a CSV file (Housing.csv) containing real estate data. The features include:

price: The price of the house (target variable).

area: The area of the house in square feet.

bedrooms: The number of bedrooms.

bathrooms: The number of bathrooms.

stories: The number of floors/stories in the house.

mainroad: Whether the house is located on the main road (yes or no).

guestroom: Whether the house has a guestroom (yes or no).

basement: Whether the house has a basement (yes or no).

hotwaterheating: Whether the house has hot water heating (yes or no).

airconditioning: Whether the house has air conditioning (yes or no).

parking: The number of parking spaces available.

prefarea: Whether the house is located in a preferred area (yes or no).

furnishingstatus: Whether the house is furnished, semi-furnished, or unfurnished.

Project Steps:
Data Loading and Inspection: The data is loaded using Pandas, and basic information like the head, info, and statistical summary of the dataset is explored.

Data Preprocessing: Missing values and categorical variables are handled. Label encoding is used to convert categorical features into numerical ones.

Data Exploration:

Distribution plots for numerical features.

Box plots to detect outliers in the data.

Count plots and bar plots for categorical features.

Modeling:

The dataset is split into training and testing sets (80-20 split).

Three machine learning models are used: Linear Regression, Decision Tree Regressor, and Random Forest Regressor.

Model evaluation metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score are used to evaluate the model's performance.

Model Evaluation: The performance of each model is compared based on MAE, RMSE, and R² Score. Results are visualized using bar plots.

Results:
The following models were evaluated:

Linear Regression: MAE = 979,679.69, R² = 0.6495

Decision Tree: MAE = 1,222,399.08, R² = 0.4682

Random Forest: MAE = 1,025,289.68, R² = 0.6115

The Linear Regression model performed the best among the three.

Future Improvements:
Hyperparameter Tuning: Perform hyperparameter optimization for the decision tree and random forest models.

Additional Features: Include more features like location, neighborhood, etc., for better predictions.

Deployment: Create a web application or API to allow users to input house features and get price predictions.

How to Run:
Install the necessary libraries:

bash
Copy
Edit
pip install pandas matplotlib seaborn scikit-learn
Run the Python script:

bash
Copy
Edit
python real_estate_price_prediction.py
The output will include:

Data exploration details.

Model evaluation results.

Visualizations comparing the model performances.
