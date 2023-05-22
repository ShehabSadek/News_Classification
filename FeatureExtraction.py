from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors


def n_gram(phrase,n):
    word_tokens = word_tokenize(phrase)
    ngram_features = list(ngrams(word_tokens, n))
    ngram_features = [' '.join(gram) for gram in ngram_features]
    return ngram_features

def tf_idf(phrases, n):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(n, n))
    tfidf_matrix = tfidf_vectorizer.fit_transform(phrases)
    tfidf_features = tfidf_matrix.toarray()
    return tfidf_features


def word2vec():
    # 7ad y3mlha i give up
    return