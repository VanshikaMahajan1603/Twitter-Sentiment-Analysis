
## Twitter Sentiment Analysis using Machine Learning

### Overview 

**Situation**
Social media, especially Twitter, is a powerful platform for public expression. Understanding public sentiment from tweets is valuable for businesses, policymakers, and researchers—but tweets are short, informal, and noisy, making them hard to analyze using traditional methods.

**Task**
The project aimed to build a reliable and scalable sentiment analysis system that classifies tweets as **positive** or **negative**. The goal was to handle noisy text, ensure high accuracy, and deploy a usable interface for real-time sentiment prediction.

**Action**

* **Data Preparation**: A balanced subset of 212,000 tweets was taken from the TSATC dataset and split into training, validation, and test sets.
* **Preprocessing**: Tweets were cleaned using `neattext` (removing mentions, special characters, and stopwords).
* **Feature Engineering**: Applied **TF-IDF** and **CountVectorizer** to convert text into numerical features.
* **Model Training**: Trained four classifiers—**Logistic Regression, SVM, Random Forest, and Perceptron**—and evaluated them using accuracy, precision, recall, F1 score, and ROC-AUC.
* **Deployment**: Built a real-time sentiment prediction web app using Streamlit, allowing users to input tweets and see sentiment predictions with confidence scores and visualizations.

**Result**

* **Logistic Regression** emerged as the best model with **76.2% accuracy and 0.81 AUC**.
* A fully functional web app was created for live tweet analysis.
* ROC and confusion matrices validated performance across models.
* The system offers meaningful insights for brands, researchers, and policymakers to understand public mood instantly.

---

### Features

* Balanced dataset of 212,000 tweets for unbiased training
* TF-IDF and CountVectorizer for effective text representation
* Multiple ML algorithms compared with full evaluation metrics
* Robust text cleaning using `neattext`
* Real-time prediction app using Streamlit with visual feedback
* Saved trained pipeline for future use (`.pkl` file)

---

### Future Enhancements

* Integrate deep learning models like LSTM or transformer-based models (e.g., BERT) for contextual understanding.
* Expand to multilingual sentiment analysis.
* Improve sarcasm and emotion detection using attention mechanisms.
* Enable real-time tweet monitoring from Twitter APIs.
* Explore ensemble models for enhanced performance.

---

### Tech Stack

| Category           | Tools/Technologies                       |
| ------------------ | ---------------------------------------- |
| Language           | Python                                   |
| Libraries          | scikit-learn, pandas, numpy, neattext    |
| ML Models          | Logistic Regression, SVM, Perceptron, RF |
| Feature Extraction | TF-IDF, CountVectorizer                  |
| Visualization      | matplotlib, seaborn, altair              |
| Deployment         | Streamlit, joblib                        |

---
### Dataset Used
https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis


