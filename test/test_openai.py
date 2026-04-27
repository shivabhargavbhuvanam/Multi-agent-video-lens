import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Say hello"}]
)
print(response.choices[0].message.content)