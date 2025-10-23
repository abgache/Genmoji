import google.generativeai as genai
import requests

def gemini_ai(prompt: str, api_keya: str):
    genai.configure(api_key=api_keya)
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(prompt)
    return response.text

def local_ai(prompt: str, model: str = "llama3.1:8b", port=11434, error=True):
    url = f"http://localhost:{str(port)}/api/generate"
    headers = {
        "Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False}
    try:
        response = requests.post(url, headers=headers, json=data)
    except ConnectionError:
        import subprocess as sub
        sub.run("ollama run llama3.1:8b")

    if response.status_code == 200:
        return response.json()["response"]
    else:
        if error:
            raise Exception(f"Ollama Error: {response.status_code} - {response.text}\nMake sure your ollama is running, that the right port as given, and that LLaMa3.1:8b is downloaded.")
        else:
            return None

if __name__ == "__main__": 

    print(f"User: Explain how Artificial Intelligence works.\nLLaMa:\n{local_ai('Explain how Artificial Intelligence works.')}\nGemini:\n{gemini_ai('Explain how Artificial Intelligence works.', api_keya=input("Please enter your google API key > "))}")
