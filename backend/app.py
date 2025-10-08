from Flask import Flask, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)



if __name__ == "__main__":
    app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
