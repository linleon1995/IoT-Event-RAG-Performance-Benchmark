import random
from datetime import datetime, timedelta

EVENT_NAMES = ["Fire Alarm", "Unauthorized Access", "Water Leak", "Power Outage"]
LOCATIONS = ["Building A", "Building B", "Warehouse", "Lobby"]
DEVICES = ["Camera_1", "Sensor_A", "Drone_X", "Device_42"]

def generate_event(index):
    time = datetime.now() - timedelta(minutes=random.randint(0, 10000))
    return {
        "time": time.strftime("%Y-%m-%d %H:%M"),
        "location": random.choice(LOCATIONS),
        "event_name": random.choice(EVENT_NAMES),
        "device_name": random.choice(DEVICES),
        "video": f"video_{index}.mp4"
    }

def generate_events(n=20):
    return [generate_event(i) for i in range(n)]
