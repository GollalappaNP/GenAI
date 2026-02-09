# Module or library install command (run this in terminal before running the script)
# pip install gensim scipy

# Import required libraries
import gensim.downloader as api  # For downloading pre-trained word vectors
from scipy.spatial.distance import cosine  # For calculating cosine similarity

# Load pre-trained Word2Vec model (Google News, 300 dimensions)
print("Loading Word2Vec model...")
model = api.load("word2vec-google-news-300")
print("Model loaded successfully.\n")

# Get and print the first 10 dimensions of the word vector for 'king'
vector = model['king']
print("First 10 dimensions of 'king' vector:")
print(vector[:10], "\n")

# Print top 10 most similar words to 'king'
print("Top 10 words most similar to 'king':")
for word, similarity in model.most_similar('king'):
    print(f"{word}: {similarity:.4f}")
print()

# Perform word analogy: king - man + woman ≈ queen
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print("Analogy - 'king' - 'man' + 'woman' ≈ ?")
print(f"Result: {result[0][0]} (Similarity: {result[0][1]:.4f})\n")

# Analogy: paris + italy - france ≈ rome
print("Analogy - 'paris' + 'italy' - 'france' ≈ ?")
for word, similarity in model.most_similar(positive=['paris', 'italy'], negative=['france']):
    print(f"{word}: {similarity:.4f}")
print()

# Analogy: walking + swimming - walk ≈ swim
print("Analogy - 'walking' + 'swimming' - 'walk' ≈ ?")
for word, similarity in model.most_similar(positive=['walking', 'swimming'], negative=['walk']):
    print(f"{word}: {similarity:.4f}")
print()

# Calculate cosine similarity between 'king' and 'queen'
similarity = 1 - cosine(model['king'], model['queen'])
print(f"Cosine similarity between 'king' and 'queen': {similarity:.4f}")