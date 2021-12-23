import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
tokenizer = T5Tokenizer.from_pretrained('t5-small')
print("loaded tokenizer")
model = T5ForConditionalGeneration.from_pretrained('t5-small')
print("loaded model")
model.load_state_dict(torch.load('./save_model_T52021-12-20 18 30 54.119469.pt'))
print("loaded weights")

@app.route('/')
def home():
    return render_template('index.html')


def infer(aspect, text):
    try:
        aspect_prefix = "aspect: "
        sentence_prefix = " sentence: "
        max_source_length = 512

        tokenized_input = tokenizer(
            aspect_prefix + aspect + sentence_prefix + text,
            padding='longest',
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt")

        model_outputs = model.generate(tokenized_input['input_ids'])
        output_text = tokenizer.decode(model_outputs[0], skip_special_tokens=True).lower()
        if output_text.find("positive") != -1:
            return "positive"
        elif output_text.find("negative") != -1:
            return "negative"
        else:
            return "neutral"
    except:
        return "Error"


@app.route('/predict', methods=['POST'])
def predict():
    aspect = request.form.get('aspect')
    text = request.form.get('text')

    output = infer(aspect, text)

    return render_template('index.html', prediction_text='Aspect Sentiment is: {}'.format(output))


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)
