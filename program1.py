# Module or library install command (run this in terminal before running the script)
# pip install sentence-transformers scipy nltk

import nltk
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

# Download NLTK words corpus (first run only)
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

from nltk.corpus import words

# Load sentence transformer model (compatible with Python 3.14)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.\n")


def get_vector(word):
    """Get embedding vector for a word."""
    return model.encode([word], convert_to_numpy=True)[0]


def most_similar(positive=None, negative=None, topn=10):
    """Find most similar words. Supports single word or analogy (positive/negative)."""
    if positive is None:
        positive = []
    if negative is None:
        negative = []

    # Build target vector
    if positive and not negative:
        # Single word: most similar to this word
        target = get_vector(positive[0])
        exclude = set(w.lower() for w in positive)
    else:
        # Analogy: positive[0] + positive[1] - negative[0]
        target = np.zeros(model.get_sentence_embedding_dimension())
        for w in positive:
            target += get_vector(w)
        for w in negative:
            target -= get_vector(w)
        exclude = set(w.lower() for w in positive + negative)

    # Use vocabulary (common English words, limit for speed)
    word_list = list(dict.fromkeys(w.lower() for w in words.words() if 2 <= len(w) <= 12 and w.isalpha()))[:50000]
    word_list = [w for w in word_list if w not in exclude]

    # Encode in batches for efficiency
    embeddings = model.encode(word_list, convert_to_numpy=True, show_progress_bar=False)

    # Cosine similarity (1 - cosine distance)
    target_norm = target / (np.linalg.norm(target) + 1e-10)
    similarities = np.dot(embeddings, target_norm)

    # Get top N
    top_indices = np.argsort(similarities)[::-1][:topn]
    return [(word_list[i], float(similarities[i])) for i in top_indices]


# Get and print the first 10 dimensions of the word vector for 'king'
vector = get_vector('king')
print("First 10 dimensions of 'king' vector:")
print(vector[:10], "\n")

# Print top 10 most similar words to 'king'
print("Top 10 words most similar to 'king':")
for word, similarity in most_similar(positive=['king'], topn=10):
    print(f"{word}: {similarity:.4f}")
print()

# Perform word analogy: king - man + woman ≈ queen
result = most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print("Analogy - 'king' - 'man' + 'woman' ≈ ?")
print(f"Result: {result[0][0]} (Similarity: {result[0][1]:.4f})\n")

# Analogy: paris + italy - france ≈ rome
print("Analogy - 'paris' + 'italy' - 'france' ≈ ?")
for word, similarity in most_similar(positive=['paris', 'italy'], negative=['france'], topn=10):
    print(f"{word}: {similarity:.4f}")
print()

# Analogy: walking + swimming - walk ≈ swim
print("Analogy - 'walking' + 'swimming' - 'walk' ≈ ?")
for word, similarity in most_similar(positive=['walking', 'swimming'], negative=['walk'], topn=10):
    print(f"{word}: {similarity:.4f}")
print()

# Calculate cosine similarity between 'king' and 'queen'
similarity = 1 - cosine(get_vector('king'), get_vector('queen'))
print(f"Cosine similarity between 'king' and 'queen': {similarity:.4f}")
