# train_model.py
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model():
    # Încărcarea datelor de antrenament
    try:
        with open("training_data.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: training_data.json file not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON in training_data.json")
        return

    if not data:
        print("Error: No data found in training_data.json")
        return

    df = pd.DataFrame(data)

    # Check if we have the required columns
    required_columns = ['system_word', 'chosen_word', 'success']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in data")
            return

    # Crearea indexurilor pentru cuvinte
    try:
        with open("words_dictionary.json", "r") as f:
            all_words_dict = json.load(f)
        system_words = list(all_words_dict.keys())
    except FileNotFoundError:
        print("Error: words_dictionary.json file not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON in words_dictionary.json")
        return
    # system_words = list(df['system_word'].unique())
    player_words = list(df['chosen_word'].unique())

    # Mapare cuvinte la indexuri
    system_to_idx = {word: idx for idx, word in enumerate(system_words)}
    player_to_idx = {word: idx for idx, word in enumerate(player_words)}

    # Adăugare coloane cu indexuri
    df['system_idx'] = df['system_word'].map(system_to_idx)
    df['player_idx'] = df['chosen_word'].map(player_to_idx)

    # Elimină intrări cu valori lipsă
    df.dropna(subset=["system_idx", "player_idx"], inplace=True)

    # Antrenare model
    X = df[['system_idx', 'player_idx']]
    y = df['success'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Salvarea modelului și a indexurilor
    joblib.dump(model, "model.pkl")
    with open("word_indexes.json", "w") as f:
        json.dump({
            "system_words": system_words,
            "player_words": player_words
        }, f)

    print(f"Accuracy: {model.score(X_test, y_test):.2f}")


if __name__ == "__main__":
    train_model()
