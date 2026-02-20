# Required Libraries:
# Install gensim if not already installed:
# pip install gensim

import gensim.downloader as api  # For loading pre-trained word embeddings
from gensim.models import KeyedVectors  # For working with word vectors
import random  # For shuffling similar words

# Load pre-trained GloVe word vectors (100-dimensional, trained on Wikipedia + Gigaword)
model = api.load("glove-wiki-gigaword-100")

# Function to generate similar words for a given seed word
def generate_similar_words(seed_word, topn=10):
    # Check if the seed word exists in the model vocabulary
    if seed_word in model:
        # Return top 'n' similar words based on cosine similarity
        return [word for word, _ in model.most_similar(seed_word, topn=topn)]
    else:
        # Return empty list if word not in vocabulary
        return []

# Function to create a meaningful paragraph using the seed and its similar words
def create_paragraph(seed_word):
    similar_words = generate_similar_words(seed_word, topn=10)
    if not similar_words:
        return f"No similar words found for '{seed_word}'."

    # Randomly shuffle similar words and select 5
    random.shuffle(similar_words)
    selected_words = similar_words[:5]

    # Construct a short creative paragraph
    paragraph = f"In a world defined by {seed_word}, "
    paragraph += f"people found themselves surrounded by concepts like {', '.join(selected_words[:-1])}, and {selected_words[-1]}. "
    paragraph += f"These ideas shaped the way they thought, acted, and dreamed. Every step forward in their journey reflected the essence of '{seed_word}', "
    paragraph += f"bringing them closer to understanding the true meaning of {selected_words[0]}."

    return paragraph

# Example usage
seed = "freedom"  # You can change this to any word like 'love', 'innovation', etc.
print(create_paragraph(seed))
