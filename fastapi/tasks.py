from time import sleep
import random


def generate_report(report_type: str):
    """Simulate a report that takes real time to generate."""
    durations = {
        "sales": 8,
        "inventory": 5,
        "analytics": 12,
    }
    duration = durations.get(report_type, 6)
    sleep(duration)
    return {
        "report_type": report_type,
        "rows": random.randint(100, 5000),
        "status": "complete",
    }
