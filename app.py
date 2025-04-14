import torch
import streamlit as st
import requests
from transformers import BertTokenizer
from bert_modules import BERTForClassification, BERTLM, BERT, AGNewsDataset
from newsapi import NewsApiClient

device = "cpu"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BERT(len(tokenizer.vocab)).to(device)
dataset = AGNewsDataset(tokenizer=tokenizer, seq_len=128)

# Initialize BERT LM model and classification model
pretrained_bert_lm = BERTLM(bert_model, len(tokenizer.vocab)).to(device)
pretrained_bert_lm.load_state_dict(
    torch.load("bert_model_wiki103.pth", map_location=torch.device('cpu'))
)

# Extract encoder weights and initialize BERT encoder
bert_state_dict = {
    key.replace("bert.", ""): value
    for key, value in pretrained_bert_lm.state_dict().items()
    if key.startswith("bert.")
}

bert_encoder = BERT(vocab_size=len(tokenizer.vocab)).to(device)
bert_encoder.load_state_dict(bert_state_dict)

# Load the classification model
classification_model = BERTForClassification(bert_encoder, num_classes=4).to(device)
classification_model.load_state_dict(
    torch.load("best_ft_model.pth", map_location=torch.device('cpu'))
)
classification_model.eval()

# Define category labels
labels = ["World", "Sports", "Business", "Sci/Tech"]

# Streamlit UI setup
st.title("AI-Powered News Recommender")
user_input = st.text_input("Enter desired category (World, Sports, Business, Sci/Tech):", "").strip().lower()

# Fetch and recommend articles when button is clicked
if st.button("Fetch & Recommend"):
    if not user_input:
        st.error("Please enter a valid category.")
    else:
        with st.spinner("Fetching news articles..."):
            # Fetch top headlines using News API
            newsapi = NewsApiClient(api_key="84cf943dc11e4bf8b1f266048427ffa5")
            top_headlines = newsapi.get_top_headlines(language="en", page_size=20)
            articles = top_headlines.get("articles", [])

            recommendations = []

            # Process articles and predict categories
            for article in articles:
                title = article.get("title", "")
                url = article.get("url", "")

                if title:
                    # Process the article title with tokenizer and model
                    processed = dataset.from_text(title)
                    input_data = {k: v.unsqueeze(0).to(device) for k, v in processed.items()}
                    with torch.no_grad():
                        logits = classification_model(input_data["bert_input"], input_data["segment_label"])
                        pred_class = torch.argmax(logits, dim=1).item()
                        predicted_category = labels[pred_class].lower()

                        # Filter articles based on predicted category
                        if predicted_category == user_input:
                            recommendations.append((title, predicted_category, url))

        # Display results
        if recommendations:
            st.success(f"Found {len(recommendations)} articles in '{user_input.capitalize()}' category")
            for title, category, url in recommendations:
                st.markdown(f"##### [{title}]({url})")
                st.write(f"**Predicted Category:** {category.capitalize()}")
        else:
            st.warning("No matching articles found.")
