import os
import time
from subprocess import Popen, PIPE
from flask import Flask, render_template, request
from multiprocessing.pool import ThreadPool

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_DIR = os.path.join(BASE_DIR, "llama.cpp")

# Initialize the llama process
llama_process = Popen([
    os.path.join(LLAMA_DIR, "main"),
    "-m", os.path.join(LLAMA_DIR, "models/7B/ggml-model-q4_0.bin"),
    "-t", "8", "-n", "128"
], stdin=PIPE, stdout=PIPE, stderr=PIPE)

# Initialize the thread pool
pool = ThreadPool(processes=10)

# Cache the responses for 10 seconds
response_cache = {}
CACHE_TIMEOUT = 10

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form.get("prompt")
        return render_template("index.html", processed_text=llama(prompt))
    
    return render_template("index.html")

def llama(prompt):
    # Check if the response is cached
    if prompt in response_cache:
        if time.time() - response_cache[prompt]['timestamp'] < CACHE_TIMEOUT:
            return response_cache[prompt]['response']
        else:
            del response_cache[prompt]

    # Use the pre-warmed llama process
    p = llama_process

    # Run the llama process asynchronously
    async_result = pool.apply_async(p.communicate, (prompt.encode('utf-8'),))

    # Wait for the response
    stdout, stderr = async_result.get()

    # Cache the response
    response_cache[prompt] = {
        'response': "|".join(stdout.decode("utf-8").splitlines()),
        'timestamp': time.time()
    }

    return response_cache[prompt]['response']

if __name__ == '__main__':
    app.run()
