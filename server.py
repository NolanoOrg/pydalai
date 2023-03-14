from flask import Flask, render_template, request
from subprocess import Popen, PIPE
import sys
sys.path.append("./llama.cpp/")



app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form.get("prompt")
        return render_template("index.html", processed_text=llama(prompt))
    
    return render_template("index.html")

def llama(prompt):
    p = Popen(["./llama.cpp/main", "--prompt", prompt, "-m", "./llama.cpp/models/7B/ggml-model-q4_0.bin", "-t", "8", "-n", "128"], stdout=PIPE, stderr=PIPE)

    lines = p.stdout.readlines()

    # join the lines using | as a separator
    return "|".join([line.decode("utf-8").strip() for line in lines])


def do_something_with_text(text):
    return "This is a test string|It works"

if __name__ == '__main__':
    app.run()