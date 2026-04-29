import sys
import os

sys.path.append(os.path.abspath("src"))

from predict import predict_anomaly


def run_prediction(sequence):
    return predict_anomaly(sequence)