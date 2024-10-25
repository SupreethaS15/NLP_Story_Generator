from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Sample training data for genre classification
train_data = [
    ('A brave explorer discovers hidden treasures', 'adventure'),
    ('A detective solves a complex mystery', 'mystery'),
    ('A young hero wields magic to save the kingdom', 'fantasy'),
    ('The group was trapped in a haunted house', 'horror'),
    ('The astronauts explore a new planet', 'sci-fi'),
    ('A dark curse threatens the kingdom', 'fantasy'),
    ('A dangerous alien attack on the space station', 'sci-fi'),
    ('Lost in the jungle with dangerous creatures', 'adventure'),
    ('An unsolved mystery leaves the town in fear', 'mystery'),
    ('Survivors escape from a nightmare scenario', 'horror'),
    # Add more diverse data here
    ('A lost city filled with danger and excitement', 'adventure'),
    ('A chilling tale of a vampire haunting a small town', 'horror'),
    ('A magical realm where anything is possible', 'fantasy'),
    ('Aliens invade Earth and humanity fights back', 'sci-fi'),
    ('A complex plot involving espionage and betrayal', 'mystery'),
    ('A group of friends searching for a hidden treasure', 'adventure'),
    ('A haunting presence in an abandoned mansion', 'horror'),
    ('A young wizard’s journey to master magic', 'fantasy'),
    ('Exploring the cosmos on a daring mission', 'sci-fi'),
    ('Detectives piece together clues to solve a murder', 'mystery'),
    ('An explorer ventures deep into the Amazon rainforest', 'adventure'),
    ('A cursed object brings misfortune to its owner', 'horror'),
    ('A quest for a powerful artifact to save the realm', 'fantasy'),
    ('A future where robots rule and humans fight back', 'sci-fi'),
    ('An old diary reveals secrets of a town’s dark past', 'mystery')
]

# Vectorize the training data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
train_features = tfidf_vectorizer.fit_transform([text for text, label in train_data])
train_labels = [label for text, label in train_data]

# Train an SVM classifier
genre_classifier = SVC(kernel='linear')
genre_classifier.fit(train_features, train_labels)

# Sample test data
X_test = [
    'A thrilling chase through a dark alley',  # adventure
    'A ghost story that chills the heart',      # horror
    'Exploring ancient ruins with magical beings', # fantasy
    'A spaceship lands on an unknown planet',    # sci-fi
    'Detectives uncover secrets in the old town'  # mystery
]

# Transform test data into feature vectors using the same TF-IDF vectorizer
X_test_features = tfidf_vectorizer.transform(X_test)

# True labels for the test data (for accuracy calculation)
y_test = ['adventure', 'horror', 'fantasy', 'sci-fi', 'mystery']

# Predict on the test set
y_pred = genre_classifier.predict(X_test_features)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Function to classify the genre of new text
def classify_genre(text):
    text_features = tfidf_vectorizer.transform([text])  # Transform input text into feature vector
    prediction = genre_classifier.predict(text_features)  # Predict the genre
    return prediction[0]  # Return the predicted genre

# Example usage of the classify_genre function
new_text = 'Magic forest '
predicted_genre = classify_genre(new_text)
print(f"The predicted genre for the text: '{new_text}' is '{predicted_genre}'.")
