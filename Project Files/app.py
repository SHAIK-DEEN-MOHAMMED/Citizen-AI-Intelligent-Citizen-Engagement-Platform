from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

FAQ = {
    "ensure ethical": "To ensure ethics and equity, Citizen AI must use transparent data collection, regular bias testing, informed consent from participants, and clear privacy measures.",
    "data privacy": "Citizen AI initiatives must follow strict data protection rules, collect only what’s needed, and store it securely with user control.",
    "bias": "Using diverse datasets, bias audits, and continual monitoring helps reduce bias in citizen-driven AI."
}

def get_answer(question):
    q = question.lower()
    for key, ans in FAQ.items():
        if key in q:
            return ans

    prompt = (
        "You are a helpful expert on Citizen AI. "
        "Answer clearly, mention data ethics, privacy, bias, and fairness.\n\n"
        f"User: {question}\n"
        "Expert:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(inputs, max_new_tokens=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def chat():
    msg = request.args.get("msg", "")
    return get_answer(msg)

if __name__ == "__main__":
    app.run(debug=True)
