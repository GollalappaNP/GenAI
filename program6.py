# Step 1: Install required libraries (only run once)
# pip install transformers torch

# Step 2: Import necessary library
from transformers import pipeline

# Step 3: Load the sentiment analysis pipeline using a pre-trained model
sentiment_pipeline = pipeline("sentiment-analysis")

# Step 4: Define input sentences (simulating real-world user reviews)
input_sentences = [
    "The new phone I bought is absolutely amazing!",
    "Worst customer service ever. I'm never coming back.",
    "The experience was average, nothing special.",
    "Fast delivery and the packaging was perfect.",
    "The product broke within two days. Very disappointed."
]

# Step 5: Perform sentiment analysis
results = sentiment_pipeline(input_sentences)

# Step 6: Display the results
print("Sentiment Analysis Results:\n")
for sentence, result in zip(input_sentences, results):
    print(f"Input Sentence: {sentence}")
    print(f"Predicted Sentiment: {result['label']}, Confidence Score: {result['score']:.2f}\n")