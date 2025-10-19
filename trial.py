from sklearn.feature_extraction.text import TfidfVectorizer

# Example usage
documents = [
    "This is the first document",
    "This document is the second document", 
    "And this is the third one",
    "Is this the first document?"
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Get feature names
print(vectorizer.get_feature_names_out())

# See the TF-IDF matrix
print(X.toarray())