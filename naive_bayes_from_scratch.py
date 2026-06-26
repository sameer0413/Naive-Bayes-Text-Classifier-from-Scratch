import numpy as np

class simplecountvectorizer:
    """
    A simple bag-of-words vectorizer.

    This class converts raw text into a word-count matrix.

    Example:
    documents = ["free money", "team meeting]

    vocabulary = ["free", "money", "team", "meeting"]

    Transformed matrix:
    [
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    """

    def __init__(self, lowercase=True):
        self.lowercase = lowercase
        self.vocabulary_ = {}

    def fit(self, documents):
        """
        Build the vocabulary from training documents.

        Parameters
        ------------
        documents : list of str
            The raw text documents used for training.
        """
        self.vocabulary_ = {}

        for document in documents:
            tokens = self._tokenize(document)

            for token in tokens:
                if token not in self.vocabulary_:
                    self.vocabulary_[token] = len(self.vocabulary_)

        return self

    def transform(self, documents):
        """
        Convert documents into a word-count matrix.

        Rows represent documents.
        Columns represent vocabulary words.
        Values represent word counts.
        """
        n_documents = len(documents)
        n_words = len(self.vocabulary_)

        x = np.zeros((n_documents, n_words), dtype=float)

        for row_index, document in enumerate(documents):
            tokens = self._tokenize(document)

            for token in tokens:
                column_index = self.vocabulary_[token]
                x[row_index, column_index] += 1

        return x

    def fit_transform(self, documents):
        """
        Build the vocabulary and transform documents in one step.
        """
        self.fit(documents)
        return self.transform(documents)

    def _tokenize(self, document):
        """
        Convert one document into a list of tokens.

        This simple tokenizer:
        - optionally lowercases test
        - splits text by spaces
        - removes empty tokens
        """
        if self.lowercase:
            document = document.lower()

        tokens = document.split()

        tokens = [token for token in tokens if token.strip() != ""]

        return tokens

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier implemented with NumPy only.

    This model is commonly used for text classification.

    It learns:
    -log prior probabilities for each class
    -log likelihood probabilities for each word given each class

    Prediction is made by choosing the class with the highest log score.
    """

    def __init__(self, alpha=1.0):
        """
        Parameters
        -----------
        alpha : float
            Laplace smoothing value.
            alpha=1.0 means add-one smoothing.
        """
        if alpha < 0:
            raise ValueError("alpha must be greater than or equal to 0.")

        self.alpha = alpha
        self.classes_ = None
        self.log_prior_ = None
        self.log_likelihood_ = None

    def fit(self, x, y):
        """
        Train the Naive Bayes classifier.

        Parameters
        -----------
        x: array-like of shape (n_samples, n_features)
            Word-count matrix.

        y : array-like of shape (n_samples,)
            Class labels.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

        self.classes_, class_counts = np.unique(y, return_counts=True)

        self.log_prior_ = np.log(class_counts / len(y))

        log_likelihoods = []

        for class_label in self.classes_:
            x_class = x[y == class_label]

            word_counts = x_class.sum(axis=0)

            total_word_count = word_counts.sum()

            vocabulary_size = x.shape[1]

            smoothed_word_counts = word_counts + self.alpha

            smoothed_total_count = (
                total_word_count + self.alpha * vocabulary_size
            )
            word_probabilities = (
                smoothed_word_counts / smoothed_total_count
            )

            log_likelihoods.append(np.log(word_probabilities))

        self.log_likelihood_ = np.array(log_likelihoods)

        return self

    def predict(self, x):
        """
        Predict class labels for new documents.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Word-count matrix for new documents.
        """
        x = np.asarray(x, dtype=float)

        scores = x @ self.log_likelihood_.T + self.log_prior_

        best_class_indices = np.argmax(scores, axis=1)

        return self.classes_[best_class_indices]

    def predict_log_proba(self, x):
        """
        Return raw log scores for each class.

        These are not normalized probabilities.
        They are useful for understanding how the model makes decisions.
        """
        x = np.asarray(x, dtype=float)

        scores = x @ self.likelihood_.T + self.log_prior_

        return scores