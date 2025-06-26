import random
from datetime import datetime, timedelta

from langchain_core.documents import Document

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


def make_documents(events):
    docs = []
    for idx, e in enumerate(events):
        # Determine ordinal suffix for idx+1
        n = idx + 1
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        ordinal = f"{n}{suffix}"

        content = (
            f"This is the {ordinal} event. "
            f"At {e['time']}, the device '{e['device_name']}' located at '{e['location']}' "
            f"detected the event: '{e['event_name']}'. "
            f"Video evidence is available at: {e['video']}."
        )
        metadata = {
            "time": e['time'],
            "location": e['location'],
            "event_name": e['event_name'],
            "device_name": e['device_name'],
            "video_link": e['video'],
            "event_id": f"event_{idx}" # Add a unique ID for graph nodes
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs