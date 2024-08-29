import streamlit as st
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
RawData = pd.read_csv("story1.csv", encoding='latin-1')

# Preprocess data
Data = RawData[['Story', 'Keyword']]
Data = pd.concat([Data] * 100, ignore_index=True)
Data = Data[pd.notnull(Data['Story'])]

# Clean text function
def clean_sentence(sentence):
    sentence = re.sub(r'\d+', '', sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = sentence.lower()
    sentence = sentence.strip()
    sentence = re.sub(r'x{2,}', '', sentence)
    return sentence

# Clean the 'Story' column
Data['clean'] = Data['Story'].apply(clean_sentence)

# Label encoding for the 'Keyword' column
label_encoder = LabelEncoder()
label_encoder.fit(Data['Keyword'])
Data['Keyword_encoded'] = label_encoder.transform(Data['Keyword'])

# Feature extraction using TF-IDF
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
tfidf_vect.fit(Data['clean'])
xtrain_final = tfidf_vect.transform(Data['clean'])

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=400, C=1.0).fit(xtrain_final, Data['Keyword_encoded'])

# Streamlit UI
st.title("Story Moral Predictor")

story = st.text_area("Enter your story here:")

if st.button("Get Moral & Title"):
    if story:
        cleaned_story = clean_sentence(story)
        story_features = tfidf_vect.transform([cleaned_story])

        # Make a prediction
        prediction = lr_model.predict(story_features)
        decoded_labels = label_encoder.inverse_transform(prediction)

        # Fetch a relevant moral
        moral = None
        if len(decoded_labels) > 0:
            relevant_morals = RawData[RawData['Keyword'] == decoded_labels[0]]['Moral']
            moral = relevant_morals.iloc[0] if not relevant_morals.empty else "No moral found."

        # st.write(f"**Keyword:** {decoded_labels[0]}")
        st.write(f"**Moral:** {moral}")

        from transformers import T5ForConditionalGeneration, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        def preprocess_input(paragraph):
            input_text = "summarize: " + paragraph
            return tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
        def generate_title(paragraph):
            input_ids = preprocess_input(paragraph)
            #print(input_ids)
            outputs = model.generate(input_ids, max_length=8,min_length=3, num_beams=4, length_penalty=2.0, early_stopping=True)
            title = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return title
        # Generate a title for the paragraph
        title = generate_title(story)
        #print(f"Generated Title: {title}")
        st.write(f"**Title:** {title}")
    else:
        st.warning("Please enter a story.")