import json
import requests
from config.env import API_KEY_HEADER, API_KEYS, DJANGO_ALLOWED_HOSTS

# API Base URL
API_BASE = f"http://{DJANGO_ALLOWED_HOSTS[0]}:8000/api"


# Headers
headers = {
    API_KEY_HEADER: API_KEYS[0],   
    "Content-Type": "application/json"
}

# Health endpoint
print("Testing health endpoint...")
response = requests.get(f"{API_BASE}/health/")
print(f"Health Check: {response.status_code}")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

# Query endpoint
print("\nTesting query endpoint...")
query_data = {
    "question": "متى ولد البيرت اينشتاين ؟"
}

response = requests.post(
    f"{API_BASE}/query/",
    headers=headers,
    json=query_data
)

print(f"Query Response: {response.status_code}")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

# Ingest endpoint
print("\nTesting ingest endpoint...")
ingest_data = {
    "text": "الذكاء الاصطناعي هو مجال من مجالات علوم الكمبيوتر يهدف إلى إنشاء أنظمة قادرة على أداء المهام التي تتطلب ذكاءً بشريًا.",
    "metadata": {
        "source": "test",
        "type": "definition"
    }
}

response = requests.post(
    f"{API_BASE}/ingest/",
    headers=headers,
    json=ingest_data
)

print(f"Ingest Response: {response.status_code}")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
