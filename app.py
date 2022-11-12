from flask import Flask
app = Flask(__name__)

from knn_parkinsons_diagnosis import call
from knn_parkinsons_diagnosis import call1


@app.route('/')
    call()
    call1()
