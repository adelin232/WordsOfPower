# best_strategy.py
from collections import defaultdict
import requests
from time import sleep
import json
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"
NUM_ROUNDS = 5
PENALTY = 30


class WordStrategy:
    def __init__(self):
        self.load_resources()
        self.text_to_id = {w['text']: idx+1 for idx,
                           w in enumerate(self.player_words)}
        self.word_usage = defaultdict(int)
        self.category_rules = {
            'natural': ['Flood', 'Earthquake', 'Tornado'],
            'weapon': ['Sword', 'Gun', 'Nuclear Bomb'],
            'animal': ['Lion', 'Whale', 'Bacteria'],
            'medical': ['Vaccine', 'Virus', 'Cure'],
            'war': ['War', 'Tank', 'Soldier'],
            'peace': ['Peace', 'Diplomacy', 'Truce']
        }
        self.safe_cost_range = (17, 32)  # Range pentru costuri sigure

    def load_resources(self):
        with open('player_words.json') as f:
            self.player_words = json.load(f)
        self.model = joblib.load('model.pkl')
        with open('word_indexes.json') as f:
            self.indexes = json.load(f)

    def get_word_category(self, word):
        word_lower = word.lower()
        for category, words in self.category_rules.items():
            if any(w.lower() == word_lower for w in words):
                return category
        return 'other'

    def predict_success(self, system_word, player_word):
        try:
            sys_idx = self.indexes['system_words'].index(system_word)
            plr_idx = self.indexes['player_words'].index(player_word)

            # Creăm un DataFrame pentru a evita avertismentul
            X = np.array([[sys_idx, plr_idx]]).reshape(1, -1)
            proba = self.model.predict_proba(X)[0][1]

            # Ajustări bazate pe categorii
            sys_cat = self.get_word_category(system_word)
            plr_cat = self.get_word_category(player_word)

            # Reguli speciale pentru 'War'
            if system_word.lower() == 'war':
                if plr_cat == 'peace':
                    return min(0.95, proba * 1.5)
                if player_word.lower() in ['peace', 'diplomacy']:
                    return 0.9

            # Reguli generale între categorii
            category_boosts = {
                ('natural', 'medical'): 1.3,
                ('animal', 'weapon'): 1.3,
                ('war', 'peace'): 1.5
            }

            boost = category_boosts.get((sys_cat, plr_cat), 1.0)
            return min(0.95, proba * boost)

        except ValueError:
            # Fallback pentru cuvinte necunoscute - alegere cost safe
            return 0.7 if self.get_word_cost(player_word) in range(*self.safe_cost_range) else 0.5

    def get_word_cost(self, word):
        return next((w['cost'] for w in self.player_words if w['text'].lower() == word.lower()), 30)

    def choose_best_word(self, system_word):
        candidates = []
        system_word_lower = system_word.lower()

        for word in self.player_words:
            word_text = word['text']
            word_cost = word['cost']
            usage_count = self.word_usage[word_text]

            # Calcul probabilitate cu ajustări
            proba = self.predict_success(system_word, word_text)

            # Penalizare pentru utilizare repetată (max 30% creștere cost)
            cost_multiplier = 1 + min(0.3, usage_count * 0.1)
            adjusted_cost = word_cost * cost_multiplier

            # Calcul cost așteptat
            expected_cost = adjusted_cost + (1 - proba) * PENALTY

            # Bonus pentru cuvinte în range-ul safe
            safe_cost_bonus = 1.2 if self.safe_cost_range[0] <= word_cost <= self.safe_cost_range[1] else 1.0

            candidates.append({
                'word': word_text,
                'expected_cost': expected_cost / safe_cost_bonus,
                'proba': proba,
                'cost': word_cost,
                'adjusted_cost': adjusted_cost
            })

        # Filtrează opțiunile slabe
        viable_candidates = [c for c in candidates if c['proba'] > 0.4]
        if not viable_candidates:
            viable_candidates = candidates

        # Sortează după: 1. cost așteptat, 2. probabilitate, 3. cost ajustat
        viable_candidates.sort(key=lambda x: (
            x['expected_cost'], -x['proba'], x['adjusted_cost']))

        # Alege din primele 3 opțiuni viable
        top_candidates = viable_candidates[:3]

        if system_word_lower == 'war':
            # Forțează alegerea unui cuvânt din categoria 'peace' dacă există
            peace_words = [c for c in top_candidates
                           if self.get_word_category(c['word']) == 'peace']
            if peace_words:
                chosen_word = peace_words[0]['word']
            else:
                chosen_word = top_candidates[0]['word']
        else:
            chosen_word = top_candidates[0]['word']

        self.word_usage[chosen_word] += 1
        return chosen_word


def play_game(player_id):
    strategy = WordStrategy()
    total_cost = 0

    for round_id in range(1, NUM_ROUNDS + 1):
        while True:
            response = requests.get(get_url)
            data = response.json()
            if data['round'] == round_id:
                sys_word = data['word']
                break
            sleep(0.5)

        # Override pentru prima rundă (test)
        if round_id == 1:
            sys_word = 'War'

        chosen_word = strategy.choose_best_word(sys_word)
        chosen_id = strategy.text_to_id[chosen_word]

        response = requests.post(
            post_url,
            json={
                "player_id": player_id,
                "word_id": chosen_id,
                "round_id": round_id
            }
        ).json()

        status = requests.get(status_url).json().get('status', {})
        cost = next(w['cost']
                    for w in strategy.player_words if w['text'] == chosen_word)

        if not status.get('p1_won', True) if 'p1_word' in status else True:
            cost += PENALTY

        total_cost += cost
        print(f"Runda {round_id}: {sys_word} -> {chosen_word} (Cost: {cost})")
        print(
            f"Probabilitate succes: {strategy.predict_success(sys_word, chosen_word):.2f}")
        print(f"Cost total acumulat: {total_cost}\n")


if __name__ == "__main__":
    play_game("UcI28jovUk")
