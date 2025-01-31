#Performing in depth EDA + Data pre-processing here (tokenization + stopword removal + lemmatization + vectorization) 
#Completing Step 1: NLP Pipeline
#Preprocess python script to maintain modularity and run directly on NLP_Minor_Project/Notebook/fake_news_nlp-pipeline.ipynb

# Import necessary functions from the preprocess.py script
from src.preprocess import load_data, basic_eda, preprocess_text, vectorize_text

# Step 1: Load the training dataset
train_df = load_data('/content/drive/MyDrive/data/fake_news/train.csv')

# Step 2: Perform in-depth EDA
# Check for missing values
print("Missing values in train dataset:\n", train_df.isnull().sum())

# Distribution of labels (real or fake news)
print("Label distribution:\n", train_df['label'].value_counts())

# Check text length distribution
train_df['text_length'] = train_df['text'].apply(lambda x: len(str(x)))
train_df['text_length'].hist(bins=50)
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Visualizing class distribution (real vs. fake news)
basic_eda(train_df)

# Step 3: Preprocess the text (Tokenization, Stopword Removal, Lemmatization/Stemming, etc.)
train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)

# Step 4: Vectorization (TF-IDF)
X_train = vectorize_text(train_df, column='cleaned_text')



