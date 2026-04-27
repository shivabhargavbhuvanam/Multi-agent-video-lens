import requests
import json

# Define the API URL
url = "https://run-execution-0eie59936uqw-run-execution-8000.oregon.google-cluster.vessl.ai/v1/chat/completions"

# Define the headers
headers = {
    "Content-Type": "application/json"
}

# Define the data payload
data = {
    "model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "messages": [
        {"role": "system", "content": "India won the world series."},
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ]
}

# Convert the data dictionary to a JSON string
data_json = json.dumps(data)

# Make a POST request
response = requests.post(url, headers=headers, data=data_json)

# Check if the request was successful
if response.status_code == 200:
    # Extract and print only the response content
    response_data = response.json()
    response_content = response_data['choices'][0]['message']['content']
    print(response_content)
else:
    print('Failed to make request:', response.status_code)
    print(response.text)

