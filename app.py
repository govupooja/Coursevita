from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import ast

app = Flask(__name__)
CORS(app)

# Load and process the dataset
df = pd.read_csv(r'C:\skill_recommendation\Job Dataset.csv')
df['User_Skills'] = df['User_Skills'].apply(lambda x: x.split(', '))  # Adjust as needed

mlb = MultiLabelBinarizer()
skills_matrix = mlb.fit_transform(df['User_Skills'])

model = NearestNeighbors(metric='cosine')
model.fit(skills_matrix)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_index = request.json.get('user_index')
    try:
        recommendations = recommend_skills(int(user_index))
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/users', methods=['GET'])
def get_users():
    users = df['User_ID'].tolist()  # Assuming 'User_ID' is the column name
    return jsonify([{'user_id': user} for user in users])

def recommend_skills(user_index, num_recommendations=3):
    distances, indices = model.kneighbors(skills_matrix[user_index].reshape(1, -1), n_neighbors=num_recommendations + 1)
    recommended_skills = set()
    for idx in indices.flatten()[1:]:
        recommended_skills.update(df['User_Skills'].iloc[idx])
    return list(recommended_skills)

if __name__ == '__main__':
    app.run(debug=True)
