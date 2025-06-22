import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the trained model
pipe_model = joblib.load("model/text_sentiment.pkl")

# Dictionary for emoji mapping
sentiment_emoji_dict = {0: "😔 Negative", 1: "😊 Positive"}


def predict_sentiment(text):
    return pipe_model.predict([text])[0]


def get_prediction_proba(text):
    return pipe_model.predict_proba([text])


def main():
    # Custom CSS styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFA22A;
        }

        .stMarkdown, h1, h2, h3, h4, h5, h6, p, label {
            color: white !important;
        }

        label[data-testid="stTextAreaLabel"] {
            color: white !important;
        }

        button {
            background-color: transparent !important;
            color: white !important;
            font-weight: bold !important;
            border: 2px solid white !important;
            border-radius: 8px !important;
        }


        .custom-title {
            background-color: white !important;
            color: black !important;
            padding: 10px;
            font-size: 20px;
            font-weight: bold;
            border-radius: 8px;
            margin-top: 20px;
            width: fit-content;
        }
        </style>
    """, unsafe_allow_html=True)

    # App Title
    st.title("Tweet Sentiment Analysis")
    st.subheader("Predict the sentiment of a tweet")

    # Input form
    with st.form(key='sentiment_form'):
        raw_text = st.text_area("Enter your tweet here:")
        submit_text = st.form_submit_button(label='Analyze')

    if submit_text:
        prediction = predict_sentiment(raw_text)
        probability = get_prediction_proba(raw_text)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="custom-title">Tweet Analysis</div>', unsafe_allow_html=True)
            st.markdown(f"{raw_text}")

            st.markdown('<div class="custom-title">Prediction</div>', unsafe_allow_html=True)
            st.markdown(f"Sentiment: {sentiment_emoji_dict[prediction]}")
            st.markdown(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.markdown('<div class="custom-title">Prediction Probability</div>', unsafe_allow_html=True)

            prob_negative = probability[0][0]
            prob_positive = probability[0][1]

            proba_df = pd.DataFrame({
                "Sentiment": ["Negative", "Positive"],
                "Probability": [prob_negative, prob_positive]
            })

            st.markdown(f"Negative Sentiment: {prob_negative:.2f}")
            st.markdown(f"Positive Sentiment: {prob_positive:.2f}")

            fig = alt.Chart(proba_df).mark_bar().encode(
                x='Sentiment',
                y='Probability',
                color='Sentiment'
            )

            st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()




# streamlit run model/Run_streamlit.py
