import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the top 5 words
def predict_top_words(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    
    # Get top 5 word indices and their probabilities
    top_indices = np.argsort(predicted[0])[::-1][:5]
    top_probabilities = np.sort(predicted[0])[::-1][:5]
    
    # Map indices to words
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    top_words = [(index_word.get(index), prob) for index, prob in zip(top_indices, top_probabilities) if index_word.get(index) is not None]
    
    return top_words

# Streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text = st.text_input("Enter the sequence of Words", "To be or not to")

if st.button("Predict Next Words"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    top_words = predict_top_words(model, tokenizer, input_text, max_sequence_len)
    
    if top_words:
        st.write("Top 5 predicted words with their probabilities:")
        for word, prob in top_words:
            # Use HTML for formatting
            st.markdown(f'<p style="font-size:20px;"><strong>{word}</strong>: <span style="font-size:14px;">{prob*100:.2f}%</span></p>', unsafe_allow_html=True)
    else:
        st.write("No predictions available.")
