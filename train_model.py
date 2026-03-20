import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    test = y[:]
    y.clear()

    for i in test:
        y.append(ps.stem(i))

    return " ".join(y)

# Download the SMS Spam Collection dataset
print("Downloading SMS Spam Collection dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print(f"Dataset loaded: {len(df)} messages")
print(f"Distribution:\n{df['label'].value_counts()}")

# Encode labels: ham=0, spam=1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Transform messages
print("\nPreprocessing messages...")
df['transformed_message'] = df['message'].apply(transform_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['transformed_message'], df['label_num'], test_size=0.2, random_state=42
)

# Create and fit TF-IDF vectorizer
print("Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train MultinomialNB model
print("Training MultinomialNB model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])}")

# Save model and vectorizer
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
print("\n[OK] Saved vectorizer.pkl and model.pkl successfully!")
print(f"  Model fitted: {hasattr(model, 'class_log_prior_')}")
