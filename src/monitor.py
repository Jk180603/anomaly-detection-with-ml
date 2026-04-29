import os
import json
import pandas as pd


def generate_monitoring_report():
    log_path = "data/logs/predictions.csv"

    if not os.path.exists(log_path):
        print("No logs found. Run predictions first.")
        return

    df = pd.read_csv(log_path)

    total_predictions = len(df)
    anomaly_count = int((df["status"] == "ANOMALY").sum())
    normal_count = int((df["status"] == "NORMAL").sum())

    anomaly_rate = anomaly_count / total_predictions if total_predictions > 0 else 0

    report = {
        "total_predictions": total_predictions,
        "normal_count": normal_count,
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(anomaly_rate, 4),
        "average_reconstruction_error": round(df["reconstruction_error"].mean(), 6),
        "max_reconstruction_error": round(df["reconstruction_error"].max(), 6),
        "min_reconstruction_error": round(df["reconstruction_error"].min(), 6),
        "latest_status": df.iloc[-1]["status"],
    }

    os.makedirs("reports", exist_ok=True)

    with open("reports/monitoring_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Monitoring Report:")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    generate_monitoring_report()