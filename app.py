import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import re

# Title and description for the Streamlit app
st.title("Food Recommendation System Based on Nutritional Values")
st.write("Enter nutritional values to get food recommendations.")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('out.csv')

df = load_data()

# Data preprocessing functions

nutritional_columns = ['nutri_kcal', 'nutri_fat', 'nutri_satuFat', 'nutri_carbohydrate', 
                       'nutri_sugar', 'nutri_fiber', 'nutri_protein', 'nutri_salt']


# Impute missing values
imputer = SimpleImputer(strategy='mean')
nutritional_data = imputer.fit_transform(df[nutritional_columns])

# Scale the data
scaler = StandardScaler()
nutritional_data_scaled = scaler.fit_transform(nutritional_data)

# KNN model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(nutritional_data_scaled)

# Recommendation function
def get_recommendations(input_nutrition, n_recommendations=8):
    input_scaled = scaler.transform(np.array(input_nutrition).reshape(1, -1))
    distances, indices = knn.kneighbors(input_scaled, n_neighbors=n_recommendations)
    recommendations = df.iloc[indices[0]]
    return recommendations[['name'] + nutritional_columns]

# Streamlit Inputs
st.subheader("Input Nutritional Values")
input_nutrition = [
    st.number_input(f"Enter value for {column}:", value=0.0) for column in nutritional_columns
]

# Display recommendations
if st.button("Get Recommendations"):
    recommendations = get_recommendations(input_nutrition)
    st.write("Recommended foods based on your input:")
    st.dataframe(recommendations) 




    # Nutritional comparison visualization
    def visualize_recommendations(input_nutrition, recommendations):
        recommendations_data = pd.concat([pd.DataFrame([input_nutrition], columns=nutritional_columns, index=['Input']),
                                          recommendations.set_index('name')[nutritional_columns]])
        recommendations_data.T.plot(kind='bar', figsize=(12, 8))
        plt.title("Nutritional Comparison Between Input and Recommended Foods")
        plt.xlabel("Nutritional Features")
        plt.ylabel("Values (Scaled)")
        plt.xticks(rotation=45)
        plt.legend(title="Foods", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot()

    visualize_recommendations(input_nutrition, recommendations)

# Correlation Heatmap
st.subheader("Correlation Heatmap of Nutritional Features")
corr_matrix = df[nutritional_columns].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
st.pyplot()