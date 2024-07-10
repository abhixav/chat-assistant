from flask import Flask, request, jsonify, render_template
import numpy as np
import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer, util
import torch
import os

app = Flask(__name__)

# Function to load dataset from JSONL file
def load_dataset(file_path):
    questions = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            questions.append(data['question'])
            answers.append(data['answer'])
    return questions, answers

# Load dataset
file_path = 'abhinav.jsonl'  # Replace with your actual file path
questions, answers = load_dataset(file_path)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
question_sequences = tokenizer.texts_to_sequences(questions)

# Pad sequences for consistent input size
max_len = max(len(seq) for seq in question_sequences)
padded_questions = pad_sequences(question_sequences, maxlen=max_len, padding='post')

# Check if the model already exists
model_path = 'model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    # Model architecture
    model = Sequential([
        Embedding(vocab_size, 16, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(len(answers), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(padded_questions, np.arange(len(answers)), epochs=500, verbose=1)

    # Save the model
    model.save(model_path)

# Load pre-trained semantic similarity model
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
corpus_embeddings = similarity_model.encode(questions, convert_to_tensor=True)

# Function to format the answer for HTML
def format_answer(answer):
    # Replace newlines with <br> and format as bullet points
    answer = answer.replace('\n', '<br>').strip()
    
    # Split the answer into lines
    lines = answer.split('<br>')
    
    # Remove any empty lines
    lines = [line for line in lines if line.strip() != '']
    
    # Join the lines into a bullet list
    formatted_answer = '<ul><li>' + '</li><li>'.join(lines) + '</li></ul>'
    
    return formatted_answer


# Function to predict and return formatted answer
def get_answer(user_input):
    try:
        # Use the semantic similarity model to find the closest question
        query_embedding = similarity_model.encode(user_input, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, corpus_embeddings)
        closest_n = torch.topk(similarities, k=1)
        closest_question_index = closest_n[1][0].item()

        # Get the answer and format it
        answer = answers[closest_question_index]
        formatted_answer = format_answer(answer)

        return formatted_answer
    except Exception as e:
        return "Sorry, I couldn't process your question."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/page1')
def page1():
    return render_template('page1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/get_answer', methods=['POST'])
def answer():
    try:
        user_input = request.json.get('question')
        predicted_answer = get_answer(user_input)
        return jsonify({'answer': predicted_answer})
    except Exception as e:
        return jsonify({'answer': "Error processing your request."})

if __name__ == '__main__':
    app.run(debug=True)
