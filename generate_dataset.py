import json
from sklearn.model_selection import train_test_split

# Load player words
with open('player_words.json', 'r') as f:
    player_words = json.load(f)

# Example dataset of word relationships (would be expanded significantly)
word_relationships = [
    {"system_word": "Tornado", "counters": [
        "Wind", "Earth", "Supernova"], "strength": 3},
    {"system_word": "Flood", "counters": [
        "Drought", "Vaccine", "Earthquake"], "strength": 2},
    {"system_word": "Hammer", "counters": [
        "Rock", "Shield", "Gun"], "strength": 1},
    # ... hundreds more examples
]

# Convert to training format
training_data = []
for rel in word_relationships:
    for counter in rel["counters"]:
        training_data.append({
            "input": rel["system_word"],
            "output": counter,
            "strength": rel["strength"]
        })

# Split into train/test sets
train_data, test_data = train_test_split(training_data, test_size=0.2)
