import spacy

# Load SpaCy's English model for chunking, POS tagging, and NER
nlp = spacy.load("en_core_web_sm")

# Function to perform POS tagging, chunking, and named entity recognition (NER)
def analyze_pos_and_chunking(text):
    doc = nlp(text)

    # POS Tagging
    print("\nPOS Tagging:")
    for token in doc:
        print(f"{token.text}: {token.pos_} ({token.dep_})")

    # Chunking (Extract Noun and Verb Phrases)
    print("\nChunking (Noun and Verb Phrases):")
    for chunk in doc.noun_chunks:
        print(f"Noun Phrase: {chunk.text}")
    for token in doc:
        if token.dep_ == "ROOT" or token.dep_ == "aux":
            print(f"Verb Phrase: {token.text}")

    # Named Entity Recognition (NER)
    print("\nNamed Entity Recognition (NER):")
    if not doc.ents:
        print("No named entities found in the text.")
    else:
        for ent in doc.ents:
            print(f"Entity: {ent.text} - Label: {ent.label_}")
