import spacy # type: ignore

# Load a model with word vectors to calculate semantic similarity
nlp = spacy.load("en_core_web_md")

# Perform semantic similarity analysis
def analyze_semantics(query_text, story_text):
    doc1 = nlp(query_text)
    doc2 = nlp(story_text)
    similarity = doc1.similarity(doc2)
    print(f"Semantic Similarity: {similarity * 100:.2f}%")
    print("-" * 30)
