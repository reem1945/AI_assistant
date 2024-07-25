from flask import Flask
from flask import  request, jsonify
import os
from vosk import Model, KaldiRecognizer
import wave
app = Flask(__name__)
import spacy
from transformers import pipeline


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

# Charger le modèle Vosk
model = Model("./vosk-model-small-fr-0.22")

def transcribe_audio(audio_content):
    wf = wave.open(audio_content, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    result = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result.append(rec.Result())
    result.append(rec.FinalResult())
    return result

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    audio_content = audio_file.read()
    transcript = transcribe_audio(audio_content)
    return jsonify({'transcript': transcript})

# Charger le modèle SpaCy pour l'analyse
nlp = spacy.load("fr_core_news_sm")

def summarize_text(text):
    # Utiliser un pipeline de résumé pré-entraîné de Hugging Face
    summarizer = pipeline("summarization",)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Exemple d'utilisation
transcribed_text = "Your transcribed text here from the Vosk model..."
summary = summarize_text(transcribed_text)
print(summary)

if __name__ == '__main__':
    app.run()
