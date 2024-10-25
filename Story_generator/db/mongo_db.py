from pymongo import MongoClient # type: ignore

# Connect to the MongoDB server
def get_db():
    client = MongoClient("mongodb://localhost:27017/")
    db = client['story_db']  # Connect to the 'story_db' database
    return db

# Fetch a document (story) based on matching keywords and classification (positive or negative)
def query_data(user_keywords, classification):
    db = get_db()
    # Match the keywords and the classification result
    return db.knowledge_base.find_one({
        "keywords": {"$in": user_keywords},
        "classification": classification
    })
