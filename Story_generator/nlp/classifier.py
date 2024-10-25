from transformers import pipeline # type: ignore

# Initialize the classifier (using a pre-trained sentiment analysis model)
classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

# Perform classification
def classify_text(text):
    result = classifier(text)[0]  # Get the classification result
    label = result['label']
    score = result['score']
    return label, score
