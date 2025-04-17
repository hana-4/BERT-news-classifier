# BERT from scratch for News Classification

This project is a category based news filtering system that leverages the **BERT model built from scratch**, pretrained on **WikiText-103**, and **fine-tuned on AG News** dataset. The app is powered by **Streamlit** and uses **NewsAPI** to fetch the latest headlines. It classifies the news articles and displays items from the category selected by the user.

![Alt text](app.png)

---

## Features

- Custom BERT architecture implemented from scratch
- Pretrained on WikiText-103 for Masked Language Modeling and Next Sentence Prediction.
- Fine-tuned on AG News for 4-class text classification:
  - World
  - Sports
  - Business
  - Sci/Tech
- Simple and interactive Streamlit UI
- Latest headline fetching using [NewsAPI](https://newsapi.org/)
- Intelligent category-based filtering and recommendation

---


## Model Pipeline

1. **Pretraining**: Masked Language Modeling and Next Sentence Prediction on WikiText-103 dataset. 
2. **Fine-tuning**: Classification on AG News dataset.
3. **Prediction**: Top headlines fetched using NewsAPI are classified, and only those matching the user’s category input are displayed.

---

## Setup Instructions


1. Clone the Repository 
```bash
git clone https://github.com/yourusername/bert-news-recommender.git
cd bert-news-recommender
```
2. Download Pretrained Weights

Download the pretrained model weights from the following Google Drive links and place them in the **project root directory**:

- [`bert_model_wiki103.pth`](https://drive.google.com/file/d/1QBsmfptkhs0e4oBciSLJWAMvTVTXzh8d/view?usp=share_link)
- [`best_ft_model.pth`](https://drive.google.com/file/d/19h8hDdYn0-wVDJc9uWBF6kuBdYu18hsr/view?usp=share_link)

>  Make sure both files are saved in the **same folder** as your `app.py`.

3. Install Dependencies
  ```bash
pip install requirements.txt 
```
4. Launch the app
```bash
streamlit run app.py 
```

