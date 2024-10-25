from nlp.story_generator import generate_story

# Main entry point
if __name__ == "__main__":
    user_input = input("Enter a theme or phrase: ")
    generate_story(user_input)
