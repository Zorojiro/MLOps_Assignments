import logging
import os
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from sklearn.linear_model import LogisticRegression
import numpy as np
from datetime import datetime

# Bonsai credentials and URL from environment variables
BONSAI_HOST = os.getenv('BONSAI_HOST')
ACCESS_KEY = os.getenv('ACCESS_KEY')
ACCESS_SECRET = os.getenv('ACCESS_SECRET')

# Set up Elasticsearch client
es = Elasticsearch(
    [{'host': BONSAI_HOST, 'port': 443, 'use_ssl': True}],
    http_auth=(ACCESS_KEY, ACCESS_SECRET)
)

# Create Flask application
app = Flask(__name__)

# Create index for logs if it doesn't exist
index_name = 'logs'
try:
    es.indices.create(index=index_name)
except NotFoundError:
    pass  # Index already exists
except Exception as e:
    print("Error creating index:", e)

# Custom logging handler for Elasticsearch
class ElasticSearchHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        # Prepare the log entry for Elasticsearch
        doc = {
            'timestamp': datetime.now(),
            'level': record.levelname,
            'message': log_entry,
            'service': 'my_flask_app'
        }
        # Index the log entry in Elasticsearch
        es.index(index=index_name, body=doc)

# Set up the Elasticsearch logging handler
handler = ElasticSearchHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s :: %(levelname)-8s :: %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

# Sample Logistic Regression model
model = LogisticRegression()

# Train the model on dummy data
X_train = np.array([[0, 0], [1, 1]])
y_train = np.array([0, 1])
model.fit(X_train, y_train)

@app.route('/')
def hello_world():
    app.logger.info("hello world logger")
    return 'Hello World!'

# Scikit-learn model prediction endpoint
@app.route('/predict', methods=['GET'])
def predict():
    # Get query parameters
    feature_1 = float(request.args.get('feature_1', 0))  # Default value is 0 if not provided
    feature_2 = float(request.args.get('feature_2', 0))  # Default value is 0 if not provided

    # Prepare input for the model
    input_features = np.array([[feature_1, feature_2]])

    # Make prediction
    prediction = model.predict(input_features)

    # Log the prediction
    app.logger.info(f"Prediction made: {prediction[0]} for features {input_features}")

    # Return prediction result as JSON
    return jsonify({
        'input': {
            'feature_1': feature_1,
            'feature_2': feature_2
        },
        'prediction': int(prediction[0])
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0")
