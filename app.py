import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your preprocessed dataset (encoded_data)
def load_data_and_model():
    encoded_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/encoded_data.csv')

    # Prepare X (features) and y (target)
    X = encoded_data.drop(columns=['CHURN', 'user_id'])  # Drop the target and irrelevant column
    y = encoded_data['CHURN']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier for demonstration purposes
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    return encoded_data, clf, X.columns

# Initialize the Streamlit app
def main():
    st.title('Churn Prediction App')

    # Load data and model
    churn_dataset, clf, feature_columns = load_data_and_model()

    # Create input fields for each feature
    input_data = {}
    for column in feature_columns:
        # Handle numerical features with number_input
        input_data[column] = st.number_input(f'Enter {column}', value=float(churn_dataset[column].median()))

    # Add a prediction button
    if st.button('Predict Churn'):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = clf.predict(input_df)

        # Display prediction result
        st.write('Prediction:', 'Churn' if prediction[0] == 1 else 'No Churn')

# Run the Streamlit app
if __name__ == '__main__':
    main()
