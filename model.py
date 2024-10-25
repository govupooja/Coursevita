import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import ast

# Read the CSV file
df = pd.read_csv(r'C:\skill_recommendation\Job Dataset.csv')  # Ensure this path is correct

# Print the columns to debug
print("Columns in DataFrame:", df.columns)

# Check if 'User_Skills' exists
if 'User_Skills' in df.columns:
    # Safely convert strings to lists
    df['User_Skills'] = df['User_Skills'].apply(ast.literal_eval)  # Assuming User_Skills contains list-like strings
else:
    raise KeyError("Column 'User_Skills' not found in the DataFrame.")

mlb = MultiLabelBinarizer()
skills_matrix = mlb.fit_transform(df['User_Skills'])

model = NearestNeighbors(metric='cosine')
model.fit(skills_matrix)

def recommend_skills(user_index, num_recommendations=3):
    distances, indices = model.kneighbors(skills_matrix[user_index].reshape(1, -1), n_neighbors=num_recommendations + 1)
    recommended_skills = set()
    
    for idx in indices.flatten()[1:]:
        recommended_skills.update(df['User_Skills'].iloc[idx])
    
    return list(recommended_skills)
