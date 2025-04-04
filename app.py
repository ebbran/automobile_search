from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from ir import AutomobileRetriever, prepare_data, SemanticEncoder
import logging

app = Flask(__name__, static_folder='static')
# Configure CORS with explicit settings
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize retriever during app creation
retriever = None

def create_retriever():
    """Initialize the retriever system with error handling"""
    try:
        logger.info("Initializing retriever system...")
        encoder = SemanticEncoder()
        corpus, _, doc_id_map, df = prepare_data("cars_2010_2020.csv")
        return AutomobileRetriever(corpus, encoder, doc_id_map, df)
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

# Create retriever immediately
try:
    retriever = create_retriever()
except Exception as e:
    logger.error(f"Fatal initialization error: {str(e)}")
    exit(1)

import requests

EXCHANGE_RATE_API_KEY = "a9ceafad1eb84efc5bf6a343"
BASE_CURRENCY = "USD"

def get_exchange_rate(target_currency):
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/latest/{BASE_CURRENCY}"
    response = requests.get(url).json()
    return response['conversion_rates'].get(target_currency, None)

@app.route('/convert', methods=['POST'])
def convert_currency():
    data = request.json
    amount = data.get("amount")
    target_currency = data.get("currency")

    if not amount or not target_currency:
        return jsonify({"error": "Missing parameters"}), 400

    rate = get_exchange_rate(target_currency)
    if not rate:
        return jsonify({"error": "Invalid currency"}), 400

    converted_amount = round(amount * rate, 2)
    return jsonify({"converted_amount": converted_amount, "currency": target_currency})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/search', methods=['POST'])
def search():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        query = data.get('query', '').strip()

        if len(query) < 2:
            return jsonify({"error": "Query must be at least 2 characters"}), 400

        logger.info(f"Processing search query: {query}")
        results = retriever.search(query, top_k=6)

        formatted = []
        for car_id, score, car_info in results:
            car_dict = car_info.to_dict()
            formatted_car_info = {
                "make": str(car_dict.get('Make', 'N/A')),
                "model": str(car_dict.get('Model', 'N/A')),
                "year": str(car_dict.get('Year', 'N/A')),
                "price": str(car_dict.get('Price (USD)', 'N/A')),
                "engine": str(car_dict.get('Engine Size (L)', 'N/A')) + "L",
                "fuel_type": str(car_dict.get('Fuel Type', 'N/A')),
                "transmission": "N/A",
                "horsepower": "N/A"
            }
            formatted.append({
                "car_id": car_id,
                "score": round(score, 3),
                "car_info": formatted_car_info
            })

        logger.info(f"Search results: {formatted[:2]}")
        return jsonify(formatted)

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)