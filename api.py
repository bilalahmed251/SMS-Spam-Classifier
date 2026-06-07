from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import uvicorn

# Make sure NLTK downloads are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

ps = PorterStemmer()

# Load YOUR existing trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

app = FastAPI(title="SMS Spam Classifier API")

# This is YOUR exact text transformation function from app.py
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

# Pydantic model for incoming requests
class MessageRequest(BaseModel):
    text: str
    app_source: str = "unknown"

class MessageResponse(BaseModel):
    is_spam: bool
    confidence: float
    reason: str

@app.post("/analyze", response_model=MessageResponse)
def analyze_message(request: MessageRequest):
    # 1. Preprocess the text using your function
    transformed_sms = transform_text(request.text)
    
    # 2. Vectorize the text using your tfidf
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict using your model
    # Note: If your model supports predict_proba, we can use it. 
    # For now, we will use predict.
    result = model.predict(vector_input)[0]
    
    is_spam = bool(result == 1)
    
    return MessageResponse(
        is_spam=is_spam,
        confidence=1.0 if is_spam else 0.0,
        reason="Spam Detected by ML Model" if is_spam else "Message is safe"
    )

if __name__ == "__main__":
    # Runs the server on all IP addresses (so your phone can connect to it over WiFi)
    uvicorn.run(app, host="0.0.0.0", port=8000)
