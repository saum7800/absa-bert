import numpy as np
from flask import Flask, request, jsonify, render_template
import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
import gensim.downloader
import nltk
from nltk.tokenize import word_tokenize
import string
from model import LSTMAttModel
from gensim import models

app = Flask(__name__)

model = "LSTMAtt"
if model == "T5":
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    print("loaded tokenizer")
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    print("loaded model")
    model.load_state_dict(torch.load('./save_model_T52021-12-20 18 30 54.119469.pt'))
    print("loaded weights")

elif model == "LSTMAtt":
    nltk.download('punkt')
    print("punkt downloaded")
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300', return_path=True)
    print("glove vectors stored at: ", glove_vectors)
    glove_vectors = models.keyedvectors.KeyedVectors.load_word2vec_format(
        glove_vectors, limit=500000)
    print("glove vectors loaded")
    vocab = glove_vectors.vocab
    sys_params = {'emb_dim': 300,
                  'max_sent_len_text': 125,
                  'max_sent_len_aspect': 5,
                  'str_padd': '@PADD'}
    punctuations = list(string.punctuation)
    printable = set(string.printable)

    model = LSTMAttModel(embedding_dim=300, hidden_dim=64, lstm_layer=2, dropout=0.6)
    print("model created")
    model.load_state_dict(
        torch.load('./save_model_LSTMAtt2021-12-22 06 18 16.114158.pt', map_location=torch.device('cpu')))
    print("model loaded")


def tokenize_sent(sent, text_type):
    padd = sys_params['str_padd']
    max_len = sys_params['max_sent_len_' + text_type]

    # Remove non-ascii characters
    printable = set(string.printable)
    sent = "".join(filter(lambda x: x in printable, sent)).lower()

    # Remove punctuation
    lst_tokens = [item.strip("".join(punctuations)) for item in word_tokenize(sent) if
                  item not in punctuations]

    # Strip the sentence if it exceeds the max length
    if len(lst_tokens) > max_len:
        lst_tokens = lst_tokens[:max_len]

    # Padd the sentence if the length is less than max length
    elif len(lst_tokens) < max_len:
        for j in range(len(lst_tokens), max_len):
            lst_tokens.append(padd)

    return lst_tokens


def vectorize_sent(sent):
    padd = sys_params['str_padd']
    wv_size = sys_params['emb_dim']
    padding_zeros = np.zeros(wv_size, dtype=np.float32)

    emb = []
    for tok in sent:

        # Zero-padding for padded tokens
        if tok == padd:
            emb.append(list(padding_zeros))

        # Get the token embedding from the word2vec model
        elif tok in vocab:
            emb.append(glove_vectors[tok].astype(float).tolist())

        # Zero-padding for out-of-vocab tokens
        else:
            emb.append(list(padding_zeros))

    return np.array(emb)


@app.route('/')
def home():
    return render_template('index.html')


def infer_lstmAtt(aspect, text):
    try:
        aspect = torch.tensor(np.array([vectorize_sent(tokenize_sent(aspect, "aspect"))])).float()
        text = torch.tensor(np.array([vectorize_sent(tokenize_sent(text, "text"))])).float()
        logits = model(text, aspect, [])
        preds = torch.argmax(logits, dim=1)
        if preds[0] == 0:
            return {
                "sentiment": "negative",
                "response code": 200
            }
        elif preds[0] == 1:
            return {
                "sentiment": "neutral",
                "response code": 200
            }
        elif preds[0] == 2:
            return {
                "sentiment": "positive",
                "response code": 200
            }
        else:
            return {
                "response code": 500,
                "error": "Error"
            }
    except:
        return {
            "response code": 500,
            "error": "Error"
        }


def infer_t5(aspect, text):
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
            return {
                "sentiment": "positive",
                "response code": 200
            }
        elif output_text.find("negative") != -1:
            return {
                "sentiment": "negative",
                "response code": 200
            }
        else:
            return {
                "sentiment": "neutral",
                "response code": 200
            }
    except:
        return {
            "response code": 500,
            "error": "Error"
        }


@app.route('/predict', methods=['POST'])
def predict():
    aspect = request.form.get('aspect')
    text = request.form.get('text')

    output = infer_lstmAtt(aspect, text)

    return jsonify(output)


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)
