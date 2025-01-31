#Performing in depth EDA + Data pre-processing here (tokenization + stopword removal + lemmatization + vectorization) 
#Completing Step 1: NLP Pipeline
#Preprocess python script to maintain modularity and run directly on NLP_Minor_Project/Notebook/fake_news_nlp-pipeline.ipynb

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def basic_eda(df):
    """
    Perform basic EDA:
    - Missing value check
    - Visualize class distribution
    - Visualize text length distribution
    """
    # EDA: Checking for missing values
    print("Missing values:\n", df.isnull().sum())
    
    # EDA: Visualize class distribution (real vs. fake news)
    sns.countplot(x='label', data=df)
    plt.title("Class Distribution (Real vs Fake News)")
    plt.show()

    # EDA: Check text length distribution
    df['text_length'] = df['text'].apply(lambda x: len(str(x)))
    df['text_length'].hist(bins=50)
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.show()

def preprocess_text(text):
    """
    Preprocess the text by:
    - Converting to lowercase
    - Removing unwanted characters
    - Lemmatization
    - Removing stopwords
    - Tokenizing the text
    """
    # Text cleaning: remove unwanted characters, punctuation, etc.
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    
    # Tokenization and Stopword Removal
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization (using NLTK's WordNetLemmatizer)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def vectorize_text(df, column='text'):
    """
    Convert text data to a TF-IDF representation.
    """
    vectorizer = TfidfVectorizer(max_features=10000)
    return vectorizer.fit_transform(df[column])


