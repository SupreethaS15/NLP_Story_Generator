from db.mongo_db import query_data
from nlp.syntax_analysis import analyze_syntax
from nlp.semantic_analysis import analyze_semantics
from nlp.classifier import classify_text
from nlp.genre_classifier import classify_genre
from nlp.analyze_pos_and_chunking import analyze_pos_and_chunking
from transformers import pipeline  # For using GPT-2 from Hugging Face

# Load the text generation model (GPT-2 or similar from Hugging Face)
generator = pipeline('text-generation', model='gpt2')

# Function to generate a continuation of the story based on user input
def generate_story_continuation(prompt, max_length=100):
    # Generate continuation with GPT-2
    generated_text = generator(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']
    
    # Ensure that we stop the generated story at the next full stop.
    if "." in generated_text:
        generated_text = generated_text[:generated_text.index(".") + 1]
    
    return generated_text

# Generate a story based on the user's query
def generate_story(user_input):
    # Perform sentiment classification on the user input
    sentiment_label, sentiment_score = classify_text(user_input)
    print(f"Sentiment Classification Result: {sentiment_label} (Confidence: {sentiment_score * 100:.2f}%)")
    print("-" * 30)

    # Perform genre classification on the user input
    genre = classify_genre(user_input)
    print(f"Genre Classification Result: {genre}")
    print("-" * 30)

    # Perform syntax analysis
    analyze_syntax(user_input)

    # Perform POS tagging and chunking
    analyze_pos_and_chunking(user_input)

    # Extract all words (tokens) from the user's input
    user_keywords = [word.lower() for word in user_input.split()]

    # Fetch the initial story from the database that matches any of the input keywords and the sentiment result
    story_seed = query_data(user_keywords, sentiment_label)

    if story_seed is None:
        print("No matching story found in the database.")
        return

    # Get the fetched story
    story_text = story_seed['story']
    print(f"Story Fetched from DB: {story_text}")

    # Start the story with the fetched text
    current_story = story_text

    # Loop to continue the story based on user input
    while True:
        # Prompt the user to continue the story
        user_continuation = input("Continue the story (or type 'stop' to end): ")

        if user_continuation.lower() == "stop":
            print("Story generation ended by user.")
            break

        # Perform NLP analysis on the user continuation
        print("\nPerforming NLP analysis on user continuation...\n")
        
        # Perform sentiment classification
        sentiment_label, sentiment_score = classify_text(user_continuation)
        print(f"Sentiment Classification Result: {sentiment_label} (Confidence: {sentiment_score * 100:.2f}%)")
        print("-" * 30)

        # Perform genre classification (if needed to guide the tone/style of the continuation)
        # genre = classify_genre(user_continuation)  # Uncomment if genre classification is desired per input
        # print(f"Genre Classification Result: {genre}")
        # print("-" * 30)

        # Perform syntax analysis
        syntax_analysis = analyze_syntax(user_continuation)

        # Perform POS tagging and chunking
        pos_and_chunks = analyze_pos_and_chunking(user_continuation)

        # Perform semantic similarity between the current story and user input
        analyze_semantics(user_continuation, current_story)

        # Based on the NLP analysis, you can modify the input to the model for better continuity
        # For example, you can adjust the prompt based on the sentiment or genre classification results.

        # Modify the prompt based on sentiment or other factors
        adjusted_prompt = user_continuation

        # If sentiment is negative, you may want to prompt GPT-2 with more conflict-oriented phrases.
        if sentiment_label == "negative":
            adjusted_prompt += " The tension in the air grew thicker, and a sense of unease settled over them."

        # If genre is "mystery", you can guide GPT-2 with questions or mysteries that need solving.
        if genre == "mystery":
            adjusted_prompt += " But there was still one question that haunted them: Who had been behind the disappearance all along?"

        # If genre is "romance", guide the prompt to softer emotions.
        if genre == "romance":
            adjusted_prompt += " Their eyes met, and for the first time, there was a flicker of hope that maybe things could change."

        # Generate the next part of the story using GPT-2 based on the adjusted prompt
        print("\nGenerating continuation of the story...\n")
        story_continuation = generate_story_continuation(adjusted_prompt)

        # Update the current story with the generated continuation
        current_story += " " + story_continuation

        # Display the updated story
        print(f"\nUpdated Story:\n{current_story}")
        print("-" * 30)

