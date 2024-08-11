from flask import Flask, request, jsonify
import pandas as pd
from textblob import TextBlob
from fuzzywuzzy import process
import nltk
import re
from flask_cors import CORS

nltk.download('punkt')
nltk.download('punkt_tab')
app = Flask(__name__)
CORS(app)
data = pd.read_csv('doctorg_data.csv')
data['symptom'] = data['symptom'].str.lower().str.replace(' ', '')
pivot_table = data.pivot_table(values='weight', index='name', columns='symptom', fill_value=0)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    corrected_text = str(TextBlob(text).correct())
    return corrected_text

def extract_symptoms(text, symptom_list):
    extracted_symptoms = []
    words = nltk.word_tokenize(text, language='english')
    for word in words:
        match, score = process.extractOne(word, symptom_list)
        if score > 80:
            extracted_symptoms.append(match)
    return extracted_symptoms

def predict_diseases(symptoms, pivot_table, top_n=5):
    disease_scores = {disease: 0 for disease in pivot_table.index}
    symptom_counts = {disease: 0 for disease in pivot_table.index}

    for symptom in symptoms:
        if symptom in pivot_table.columns:
            for disease in pivot_table.index:
                weight = pivot_table.at[disease, symptom]
                if weight > 0:
                    symptom_counts[disease] += 1
                    disease_scores[disease] += weight

    sorted_diseases = sorted(disease_scores.items(), key=lambda item: (symptom_counts[item[0]], item[1]), reverse=True)
    top_diseases = sorted_diseases[:top_n]
    return top_diseases
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the DoctorG API!'})

@app.route('/predict', methods=['POST'])
def predict_diseases_from_text():
    user_input = request.json.get('symptoms') 
    cleaned_input = preprocess_text(user_input)
    symptom_list = pivot_table.columns.tolist()
    extracted_symptoms = extract_symptoms(cleaned_input, symptom_list)

    if extracted_symptoms:
        top_diseases = predict_diseases(extracted_symptoms, pivot_table, top_n=5)
        if top_diseases:
            top_disease_name = top_diseases[0][0]
            top_disease_description = data.loc[data['name'] == top_disease_name, 'description'].values[0]
            cleaned_description = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s.,]', '', top_disease_description)).strip()
            paragraph_description = ' '.join(cleaned_description.split())
            return jsonify({'top_diseases': top_diseases, 'top_description': paragraph_description})
        else:
            return jsonify({'message': 'No matching diseases found.'}), 404
    else:
        return jsonify({'message': 'No matching symptoms found in the dataset.'}), 404

if __name__ == '__main__':
    app.run()
