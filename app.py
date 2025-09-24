import torch
import streamlit as st
import requests
import os
from transformers import BertTokenizer
from newsapi import NewsApiClient
import gdown

# Import from our organized modules
from src.models.bert_model import BERT, BERTLM
from src.models.classifier import BERTForClassification
from src.data.dataset import AGNewsDataset
from src.utils.config import get_config
from src.utils.logging import setup_logging, get_logger

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables only.")
    print("💡 Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

# Setup logging and configuration
logger = setup_logging(log_level="INFO")
config = get_config()

# Model file URLs and paths
url1 = 'https://drive.google.com/file/d/1QBsmfptkhs0e4oBciSLJWAMvTVTXzh8d/view?usp=sharing'
pt = config.paths.pretrained_model_path
#gdown.download(url1, pt, quiet=False, fuzzy=True)

url2 = 'https://drive.google.com/file/d/19h8hDdYn0-wVDJc9uWBF6kuBdYu18hsr/view?usp=sharing'
ft = config.paths.model_path
#gdown.download(url2, ft, quiet=False, fuzzy=True)

device = config.get_device()
logger.info(f"Using device: {device}")

# Load tokenizer and model
logger.info("Loading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(config.data.tokenizer_name)
bert_model = BERT(len(tokenizer.vocab), d_model=config.model.d_model).to(device)
dataset = AGNewsDataset(tokenizer=tokenizer, seq_len=config.model.seq_len)

# Initialize BERT LM model and classification model
logger.info("Loading pre-trained BERT model...")
pretrained_bert_lm = BERTLM(bert_model, len(tokenizer.vocab)).to(device)
pretrained_bert_lm.load_state_dict(
    torch.load(pt, map_location=torch.device(device))
)

# Extract encoder weights and initialize BERT encoder
logger.info("Extracting encoder weights...")
bert_state_dict = {
    key.replace("bert.", ""): value
    for key, value in pretrained_bert_lm.state_dict().items()
    if key.startswith("bert.")
}

bert_encoder = BERT(vocab_size=len(tokenizer.vocab), d_model=config.model.d_model).to(device)
bert_encoder.load_state_dict(bert_state_dict)

# Load the classification model
logger.info("Loading classification model...")
classification_model = BERTForClassification(bert_encoder, num_classes=config.model.num_classes).to(device)
classification_model.load_state_dict(
    torch.load(ft, map_location=torch.device(device))
)
classification_model.eval()
logger.info("Model loading completed successfully!")

# Define category labels
labels = ["World", "Sports", "Business", "Sci/Tech"]

# Streamlit UI setup
st.title("AI-Powered News Classifier")
user_input = st.text_input("Enter desired category (World, Sports, Business, Sci/Tech):", "").strip().lower()

# Fetch and recommend articles when button is clicked
if st.button("Fetch"):
    if not user_input:
        st.error("Please enter a valid category.")
    else:
        with st.spinner("Fetching news articles..."):
            # Fetch top headlines using News API
            try:
                api_key = config.api.news_api_key or os.getenv("NEWS_API_KEY")
                if not api_key:
                    st.error("NEWS_API_KEY environment variable not set")
                    st.stop()
                
                logger.info("Fetching news articles from API...")
                newsapi = NewsApiClient(api_key=api_key)
                top_headlines = newsapi.get_top_headlines(language="en", page_size=50)
                articles = top_headlines.get("articles", [])
                logger.info(f"Retrieved {len(articles)} articles")

                recommendations = []

                # Process articles and predict categories
                logger.info("Processing articles for classification...")
                for i, article in enumerate(articles):
                    title = article.get("title", "")
                    url = article.get("url", "")

                    if title:
                        try:
                            # Process the article title with tokenizer and model
                            processed = dataset.from_text(title)
                            input_data = {k: v.unsqueeze(0).to(device) for k, v in processed.items() if k in ['bert_input', 'segment_label']}
                            
                            with torch.no_grad():
                                logits = classification_model(input_data["bert_input"], input_data["segment_label"])
                                pred_class = torch.argmax(logits, dim=1).item()
                                predicted_category = labels[pred_class].lower()

                                # Filter articles based on predicted category
                                if predicted_category == user_input:
                                    confidence = torch.softmax(logits, dim=1)[0, pred_class].item()
                                    recommendations.append((title, predicted_category, url, confidence))
                                    
                        except Exception as e:
                            logger.warning(f"Failed to process article {i}: {e}")
                            continue
                
                logger.info(f"Found {len(recommendations)} matching articles")
                
            except Exception as e:
                logger.error(f"Error during article processing: {e}")
                st.error(f"An error occurred: {str(e)}")
                st.stop()

        # Display results
        if recommendations:
            # Sort by confidence score
            recommendations.sort(key=lambda x: x[3], reverse=True)
            st.success(f"Found {len(recommendations)} articles in '{user_input.capitalize()}' category")
            
            for title, category, url, confidence in recommendations[:10]:  # Show top 10
                st.markdown(f"##### [{title}]({url})")
                st.write(f"**Predicted Category:** {category.capitalize()}")
                st.write(f"**Confidence:** {confidence:.2%}")
                st.write("---")
        else:
            st.warning("No matching articles found.")


