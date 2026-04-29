import json
import os
import subprocess


def load_report():
    report_path = "reports/monitoring_report.json"

    if not os.path.exists(report_path):
        print("No monitoring report found.")
        return None

    with open(report_path, "r") as f:
        return json.load(f)


def trigger_retraining():
    print("🚀 Triggering model retraining...")

    # run training pipeline again
    subprocess.run(["python", "src/train.py"])

    print("✅ Retraining completed.")


def retraining_decision(threshold=0.3):
    report = load_report()

    if report is None:
        return

    anomaly_rate = report["anomaly_rate"]

    print(f"Current anomaly rate: {anomaly_rate}")

    if anomaly_rate > threshold:
        print("⚠️ High anomaly rate detected.")
        trigger_retraining()
    else:
        print("✅ System stable. No retraining needed.")


if __name__ == "__main__":
    retraining_decision()