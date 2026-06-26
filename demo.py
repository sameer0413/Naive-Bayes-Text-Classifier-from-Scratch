import numpy as np

from naive_bayes_from_scratch import (
    simplecountvectorizer,
    MultinomialNaiveBayes,
)

documents = [
    "free money now",
    "free prize money",
    "win money now",
    "team meeting today",
    "project meeting schedule",
    "team project update",
]

labels = np.array([
    "spam",
    "spam",
    "spam",
    "not_spam",
    "not_spam",
    "not_spam",
])

vectorizer = simplecountvectorizer()

x = vectorizer.fit_transform(documents)

model = MultinomialNaiveBayes(alpha=1.0)
model.fit(x, labels)

new_documents = [
    "free prize now",
    "team meeting update",
]

x_new = vectorizer.transform(new_documents)

predictions = model.predict(x_new)

print("Vocabulary:")
print(vectorizer.vocabulary_)

print("\nPredictions:")
for document, prediction in zip(new_documents, predictions):
    print(document, "->", prediction)