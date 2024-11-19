# Project Overview

This repository contains a comprehensive solution for **predictive maintenance** and **technical query handling** related to water pumps. It integrates various technologies such as **Flask**, **FAISS**, **Google Generative AI**, **Keras**, and **Sentence Transformers** to provide predictive insights and intelligent assistance.

## Backend Structure

### `app.py`

The backend is built using **Flask** and provides the following key functionalities:

#### 1. **Predictive Maintenance**
- Loads a pre-trained Keras model for sensor data analysis.
- Processes a sensor dataset (`sensor.csv`) to predict and visualize machine operation status (e.g., NORMAL, BROKEN, RECOVERING).
- Includes routes for fetching predictions and generating visualizations for sensor data.

#### 2. **Chatbot for Water Pumps**
- Reads and processes PDF files using **PyMuPDF**.
- Creates FAISS embeddings using **SentenceTransformer** for semantic similarity search.
- Leverages **Google Generative AI** to answer user queries with detailed and accurate responses.

#### 3. **Visualization**
- Generates time-series plots for sensor data filtered by day and quarter.
- Outputs plots in **base64 format** for integration with a frontend.




## Setup Instructions

### Prerequisites
1. **Python** >= 3.8
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### How to Run
1. Place the required datasets and models in the appropriate directories:
   - **Model**: `Hack_for_Good/Model_Mainntenance/sensor_model.h5`
   - **Dataset**: `Hack_for_Good/Model_Mainntenance/sensor.csv`
   - **PDF**: `Hack_for_Good/PumpO/pompe1.pdf`

2. Start the Flask server:
   ```bash
   python app.py
   ```

3. Access the following endpoints:
   - **Chatbot**: `/chatbot` (POST)
   - **Sensor Data Plot**: `/plot_sensor_data` (GET)
   - **Predictions**: `/predictions` (GET)

---

## Example Endpoints

### 1. Chatbot
- **Endpoint**: `/chatbot`
- **Method**: POST
- **Request Payload**:
  ```json
  {
    "query": "What are the main causes of water pump vibration?"
  }
  ```
- **Response**:
  ```json
  {
    "response": "Detailed explanation about water pump vibration..."
  }
  ```

### 2. Sensor Data Visualization
- **Endpoint**: `/plot_sensor_data`
- **Method**: GET
- **Parameters**:
  - `day`: (e.g., `2018-04-01`)
  - `quarter`: (1, 2, 3, or 4)
- **Response**:
  - Base64-encoded images for selected sensors.

---

### 3. Predictions
- **Endpoint**: `/predictions`
- **Method**: GET
- **Response**:
  ```json
  [
    {
      "timestamp": "04-01",
      "value": 1,
      "status": "NORMAL"
    },
    {
      "timestamp": "04-02",
      "value": 0.5,
      "status": "RECOVERING"
    }
  ]
  ```

---

## Key Features

- **Real-time Predictive Maintenance**: Detect early signs of machine failure and predict operational states.
- **Intelligent Query Handling**: Combines semantic search with Google Generative AI for detailed responses.
- **Visualization**: Interactive plots for sensor data analysis.
- **FAISS Indexing**: Efficient semantic search over PDF documents.

