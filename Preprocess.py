import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    filtered_tokens = [token for token in filtered_tokens if token.isalpha()]
    
    filtered_tokens = [token.lower() for token in filtered_tokens]
    
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    preprocessed_text = ' '.join(filtered_tokens)
    
    return preprocessed_text
