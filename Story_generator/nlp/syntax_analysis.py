import spacy # type: ignore

# Load the English model for spaCy
nlp = spacy.load("en_core_web_sm")

# Perform syntax analysis
def analyze_syntax(text):
    doc = nlp(text)
    print("Syntax Analysis:")
    for token in doc:
        print(f"{token.text}: {token.dep_}")
    print("-" * 30)
