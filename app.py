import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib

# Function to train the model
def train_model(df):
    # Preprocessing: Encode categorical variables
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_cols = encoder.fit_transform(df[['Revenue category', 'Year']])
    categories = encoder.categories_
    encoded_feature_names = []
    for i, category_list in enumerate(categories):
        encoded_feature_names.extend([f"{category_list[0]}_{category}" for category in category_list[1:]])
    df_encoded = pd.DataFrame(encoded_cols, columns=encoded_feature_names)

    # Separate features and target variable using the original DataFrame
    X = df_encoded
    y = df['Value']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # Initialize and train the Random Forest regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=10)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_predict = model.predict(X_test)
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    accuracy = model.score(X_test, y_test)

    # Save the model and encoder
    joblib.dump(model, 'pearl.joblib')
    joblib.dump(encoder, 'encoder.joblib')

    return model, encoder, mse, r2, accuracy

# Streamlit interface
st.title('Random Forest Regressor Training and Prediction Interface')

# Load the dataset
st.header('Load Dataset')

# Load the dataset directly within the code
df = pd.read_csv('cleandt.csv')
st.write(df.head(5))

# Train the model
model, encoder, mse, r2, accuracy = train_model(df)
    
st.header('Model Evaluation')
st.write(f"Mean Squared Error: {mse}")
st.write(f"R-squared: {r2}")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Prediction section
st.header('Make Predictions')
rev_category = st.selectbox('Select Revenue Category', df['Revenue category'].unique())
min_year = df['Year'].min()  # Get the minimum year from the dataset
max_year = 2025  # Set the maximum year as 2025
year = st.number_input('Enter Year', min_value=min_year, max_value=max_year, value=min_year)

if st.button('Predict'):
    # Check if the selected year is in the dataset, otherwise use the minimum year
    if year not in df['Year'].unique():
        year = min_year

    # Encode the input features
    input_df = pd.DataFrame([[rev_category, year]], columns=['Revenue category', 'Year'])

    # Ensure all categories in 'Year' are known to the encoder
    known_years = df['Year'].unique()
    input_df.loc[~input_df['Year'].isin(known_years), 'Year'] = min_year  # Replace unknown years with the minimum year

    encoded_input = encoder.transform(input_df)
    input_encoded_df = pd.DataFrame(encoded_input, columns=model.feature_names_in_)

    # Make prediction
    prediction = model.predict(input_encoded_df)
    st.write(f"Predicted Value: {prediction[0]}")

