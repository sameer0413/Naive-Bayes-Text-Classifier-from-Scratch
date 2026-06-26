## Naive Bayes Text Classifier from Scratch

This project implements a Multinomial Naive Bayes text classifier using NumPy only.

Naive Bayes is a supervised machine learning algorithm commonly used for text classification.

Examples:

- spam detection
- sentiment analysis
- document classification
- news topic classification

The model predicts the class of a document by calculating which class is most likely given the words in the document.

Example:

"free money now" → spam

"team meeting update" → not spam

### Text to Numbers

Machine learning models cannot directly understand raw text.

So the text is converted into a word-count matrix using a bag-of-words approach.

Example vocabulary:

["free", "money", "team", "meeting"]

Example document:

"free money"

Vector:

[1, 1, 0, 0]

Each number represents how many times a word appears in the document.

### Prior Probability

Prior probability means how common each class is before looking at the words.

Formula:

P(class) = number of documents in class / total number of documents

Example:

If 3 out of 6 documents are spam:

P(spam) = 3 / 6 = 0.5

The model stores this as a log probability:

log P(class)

### Likelihood Probability

Likelihood means the probability of a word appearing inside a class.

Formula:

P(word | class) =
count(word in class) / total words in class

Example:

P("free" | spam)

means:

the probability of seeing the word "free" in spam documents.

### Laplace Smoothing

Some words may not appear in a class during training.

Without smoothing, their probability would be 0.

This causes a problem because multiplying by 0 makes the whole class score 0.

Laplace smoothing fixes this by adding a small value to every word count.

Formula:

P(word | class) =
(count(word in class) + alpha)
/
(total words in class + alpha * vocabulary_size)

Usually:

alpha = 1

This is called add-one smoothing.

### Log Probabilities

Naive Bayes multiplies many probabilities together.

This can create extremely small numbers.

To avoid this, the model uses log probabilities.

Instead of multiplying:

P(class) * P(word1 | class) * P(word2 | class)

the model adds:

log P(class) + log P(word1 | class) + log P(word2 | class)

This is more numerically stable.

### Prediction

For each class, the model calculates a score:

score(class) =
log P(class)
+
sum(count(word) * log P(word | class))

The class with the highest score is selected as the prediction.

Example:

score(spam) = -4.2

score(not_spam) = -8.7

The model predicts spam because -4.2 is greater than -8.7.

### Why Naive Bayes Works

Naive Bayes works well for text because certain words are strongly associated with certain classes.

Example:

Words like "free", "win", and "prize" may appear more often in spam messages.

Words like "meeting", "project", and "schedule" may appear more often in work messages.

The model learns these word-class relationships from training data.

Then, for a new document, it combines the evidence from all words and chooses the most likely class.

### Important Note

Naive Bayes is called naive because it assumes words are independent.

This means it treats each word as if it contributes separately to the prediction.

This assumption is not always true in real language, but the model is simple, fast, and effective for many text classification tasks.