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
GOOGLE_API_KEY = "AIzaSyCKB3FGS-3NQH6FpkwfD8FUwFGrv-ijq1E"
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

# Initialize SentenceTransformer for embeddings
class GeminiEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def __call__(self, input_texts):
        # Supports batch processing
        return self.model.encode(input_texts, convert_to_tensor=False)

embed_fn = GeminiEmbeddingFunction()
dimension = 384  # Dimensionality of embeddings
index = faiss.IndexFlatL2(dimension)

# Read PDF and split into paragraphs for better FAISS indexing
def read_pdf(file_path):
    """
    Read a PDF file and extract text content with proper paragraph handling.
    
    Args:
        file_path (str): Path to the PDF file
    
    Returns:
        list: List of text paragraphs from the PDF
    """
    try:
        doc = fitz.open(file_path)
        paragraphs = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text blocks with more detailed information
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    # Process each line in the block
                    for line in block["lines"]:
                        if "spans" in line:
                            # Combine all spans in the line
                            line_text = " ".join(span["text"] for span in line["spans"])
                            if line_text.strip():
                                paragraphs.append(line_text.strip())
                
                # Add a separator between different blocks if they're not empty
                if paragraphs and paragraphs[-1]:
                    paragraphs.append("")
        
        # Clean up the paragraphs
        # Remove empty strings and join consecutive paragraphs
        cleaned_paragraphs = []
        current_paragraph = ""
        
        for p in paragraphs:
            if p.strip():
                if current_paragraph:
                    current_paragraph += " " + p.strip()
                else:
                    current_paragraph = p.strip()
            elif current_paragraph:
                cleaned_paragraphs.append(current_paragraph)
                current_paragraph = ""
        
        # Add the last paragraph if it exists
        if current_paragraph:
            cleaned_paragraphs.append(current_paragraph)
        
        # Filter out any remaining empty paragraphs and very short strings (likely artifacts)
        final_paragraphs = [p for p in cleaned_paragraphs if len(p.strip()) > 10]
        
        doc.close()
        return final_paragraphs
        
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return []

# Prepare FAISS index
documents = read_pdf('Hack_for_Good/PumpO/pompe1.pdf')
embeddings = embed_fn(documents)
embeddings = np.array(embeddings).astype(np.float32)
index.add(embeddings)

# Chatbot route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_query = request.json.get('query')
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Generate query embedding
        query_embedding = embed_fn([user_query])
        query_embedding = np.array(query_embedding).astype(np.float32)
        
        # Search FAISS index for the best match
        D, I = index.search(query_embedding, k=1)
        best_doc = documents[I[0][0]]
        
        # Generate response using Google Generative AI
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = f"""
        You are an expert in water pumps. Answer the following question based on the passage provided:
        QUESTION: {user_query}
        PASSAGE: {best_doc}
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
