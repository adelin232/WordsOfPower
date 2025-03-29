import os
import json

training_data_file = "training_data.json"

if os.path.exists(training_data_file):
    with open(training_data_file, "r") as f:
        existing_data = json.load(f)
else:
    # If the file doesn't exist yet, we'll just start an empty list
    existing_data = []

# Now you can append new rounds to existing_data:
existing_data.append({
    "system_word": "Hammer",
    "chosen_word": "Rock",
    "success": False
    # ... etc.
})

# Finally, write everything back to disk:
with open(training_data_file, "w") as f:
    json.dump(existing_data, f, indent=2)
