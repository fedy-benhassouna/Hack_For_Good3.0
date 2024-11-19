import os
import numpy as np
import pandas as pd
import base64
import io
import matplotlib
from flask import Flask, jsonify, request
from flask_cors import CORS
import faiss
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_fixed
import google.generativeai as genai
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from threading import Thread

# Use 'Agg' backend for Matplotlib to prevent issues with the main thread
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Configure Google Generative AI (use environment variable for security)
GOOGLE_API_KEY = "AIzaSyCYq-AavDWA5RhEoZ0lXBOfWzzOCf8u5dA"
genai.configure(api_key=GOOGLE_API_KEY)

# Load predictive maintenance model and dataset
model = load_model('Hack_for_Good/Model_Mainntenance/sensor_model.h5')
data = pd.read_csv('Hack_for_Good/Model_Mainntenance/sensor.csv')
data["timestamp"] = pd.to_datetime(data["timestamp"])
data['day'] = data['timestamp'].dt.strftime('%m-%d')
data.drop(['Unnamed: 0', 'sensor_15', 'sensor_50', 'sensor_51'], axis=1, inplace=True)
data.rename(columns={
    "sensor_00": "temperature",
    "sensor_04": "pression",
    "sensor_06": "debit",
    "sensor_38": "vibration"
}, inplace=True)


# Read PDF and split into paragraphs for better FAISS indexing
def read_pdf(file_path):
    try:
        # Open the PDF
        doc = fitz.open(file_path)
        print(f"Total pages in PDF: {len(doc)}")

        text = ""
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            print(f"Page {page_num} text length: {len(page_text)}")
            text += page_text

        print(f"Total text length: {len(text)}")

        # Split documents
        documents = re.split(r'\n\s*\n', text)

        # Clean documents
        documents = [
            re.sub(r'\s+', ' ', doc).strip()
            for doc in documents
            if doc.strip()
        ]

        print(f"Number of document segments: {len(documents)}")

        # Optional: Print first few documents for inspection
        for i, doc in enumerate(documents[:5], 1):
            print(f"Document {i} (length {len(doc)}):\n{doc[:200]}...\n")

        return documents

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

# Usage
# documents = read_pdf('your_file.pdf')


# Prepare FAISS index
documents = read_pdf('Hack_for_Good/PumpO/pompe1.pdf')

# Check if documents is empty and provide a useful message
if not documents:
    print("No text found in the PDF. Check if the file is valid or if the extraction method is appropriate.")

# Initialize SentenceTransformer for embeddings
class GeminiEmbeddingFunction:
    def __init__(self):
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_mode = False

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def __call__(self, input):
        # Generate embeddings for the given input text
        embeddings = self.model.encode(input, convert_to_tensor=True)
        return embeddings

embed_fn = GeminiEmbeddingFunction()

# Embed the documents individually
embeddings = []
for doc in documents:
    doc_embedding = embed_fn([doc])  # Generate embedding for each docume+nt
    embeddings.append(doc_embedding[0])  # Append each embedding to the list

# Convert embeddings to a numpy array
embeddings = np.array(embeddings)

# Check the shape of embeddings (should be (num_documents, embedding_dimension))
print("Shape of embeddings:", embeddings.shape)

# Add the embeddings to the FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for cosine similarity
index.add(embeddings)

# Chatbot route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_query = request.json.get('query')
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
   
        # Generate embedding for the query
        query_embedding = embed_fn([user_query])  # Embedding for the query
        query_embedding = np.array(query_embedding).astype(np.float32)  # Convert to numpy array

    # Search the FAISS index for the most similar document
        D, I = index.search(query_embedding, k=1)  # k=1 for top 1 document

    # Fetch the most similar document
        passage = documents[I[0][0]]  # Retrieve the most similar document

    # Format the passage and query for the prompt
        passage_oneline = passage.replace("\n", " ")

        
        # Generate response using Google Generative AI
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = f"""You are an expert in water pumps with a deep understanding of their operation, maintenance, troubleshooting, and safety protocols. \
        Your responses should provide detailed, technical information, focusing on water pump efficiency, performance optimization, \
        and preventive maintenance. Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        QUESTION: {user_query}
        PASSAGE: {passage_oneline}
        """

        response = model.generate_content(prompt)
        
        return jsonify({'response': response.text})
    
    except Exception as e:
        return jsonify({'error': str(e), 'details': 'Error processing chatbot query'}), 500

# Plot sensor data for a specific day and quarter
def plot_quarter_day_data(data, selected_day, quarter):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    filtered_data = data[data['timestamp'].dt.strftime('%Y-%m-%d') == selected_day]
    if filtered_data.empty:
        return {"error": f"No data available for the day: {selected_day}"}

    quarter_conditions = {
        1: (filtered_data['timestamp'].dt.hour >= 0) & (filtered_data['timestamp'].dt.hour < 6),
        2: (filtered_data['timestamp'].dt.hour >= 6) & (filtered_data['timestamp'].dt.hour < 12),
        3: (filtered_data['timestamp'].dt.hour >= 12) & (filtered_data['timestamp'].dt.hour < 18),
        4: (filtered_data['timestamp'].dt.hour >= 18) & (filtered_data['timestamp'].dt.hour < 24)
    }
    if quarter not in quarter_conditions:
        return {"error": "Invalid quarter. Use 1, 2, 3, or 4."}

    quarter_data = filtered_data[quarter_conditions[quarter]]
    if quarter_data.empty:
        return {"error": f"No data available for Quarter {quarter} of the day: {selected_day}"}

    sensors = ["temperature", "vibration", "debit", "pression"]
    titles = ["Temperature (°C)", "Vibration (g)", "Debit (L/min)", "Pression (kPa)"]
    colors = ["red", "blue", "green", "purple"]
    plot_images = {}
    for sensor, title, color in zip(sensors, titles, colors):
        plt.figure(figsize=(10, 6))
        plt.plot(quarter_data["timestamp"], quarter_data[sensor], marker="o", color=color, label=title)
        plt.title(f"{title} - Quarter {quarter} of Day {selected_day}")
        plt.xlabel("Timestamp")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        plot_images[sensor] = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_images

@app.route('/plot_sensor_data', methods=['GET'])
def sensor_data_plot():
    try:
        selected_day = request.args.get('day', '2018-04-01')
        quarter = int(request.args.get('quarter', 1))
        plots = plot_quarter_day_data(data, selected_day, quarter)
        return jsonify(plots)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictions', methods=['GET'])
def get_latest_predictions():
    try:
        # Your existing prediction logic
        df = data.copy()
        df['day'] = df['timestamp'].dt.strftime('%m-%d')
        
        # Create operation status
        conditions = [
            (df['machine_status'] == 'NORMAL'),
            (df['machine_status'] == 'BROKEN'),
            (df['machine_status'] == 'RECOVERING')
        ]
        choices = [1, 0, 0.5]
        df['Operation'] = np.select(conditions, choices, default=0)
        
        # Group by day and calculate mean
        daily_predictions = df.groupby('day', as_index=False)['Operation'].min()
        
        # Format for frontend
        predictions = [{
            'timestamp': row['day'],
            'value': row['Operation'],
            'status': 'NORMAL' if row['Operation'] == 1 else 'BROKEN' if row['Operation'] == 0 else 'RECOVERING'
        } for _, row in daily_predictions.iterrows()]
        
        return jsonify(predictions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
