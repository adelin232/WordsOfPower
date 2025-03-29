import json
import numpy as np
from collections import defaultdict


class EnhancedGameAI:
    def __init__(self, player_words_file='player_words.json'):
        # Încărcare cuvinte jucător
        with open(player_words_file) as f:
            self.player_words = json.load(f)

        self.cost_adjustments = defaultdict(float)

        # Construire structuri de date
        self.word_info = {w['text'].lower(): {'cost': w['cost'], 'uses': 0}
                          for w in self.player_words}
        self.all_words = [w['text'].lower() for w in self.player_words]

        # Parametri optimizați
        self.max_cost_per_round = 15
        self.learning_rate = 0.3
        self.decay_factor = 0.95

        # Inițializare model
        self.init_strategy_engine()
        self.history = []

    def init_strategy_engine(self):
        """Inițializează matricea de învățare și statistici"""
        self.relationship_matrix = defaultdict(lambda: defaultdict(float))
        self.category_clusters = self.cluster_words()
        self.cost_adjustments = defaultdict(float)

        # Inițializează relații bazate pe categorii
        for word in self.all_words:
            cluster = self.category_clusters[word]
            for other in self.all_words:
                if self.should_counter(cluster, self.category_clusters[other]):
                    self.relationship_matrix[word][other] += 0.5

    def cluster_words(self):
        """Clustering bazat pe caracteristici combinate"""
        clusters = {}
        for word in self.all_words:
            features = [
                len(word),
                self.word_info[word]['cost'],
                self.word_priority_score(word)
            ]
            # Clustering simplificat bazat pe intervale
            if features[1] < 5:
                clusters[word] = 'low_cost'
            elif features[1] > 15:
                clusters[word] = 'high_power'
            elif len(word) > 7:
                clusters[word] = 'complex'
            else:
                clusters[word] = 'neutral'
        return clusters

    def word_priority_score(self, word):
        """Calculează scorul strategic al cuvântului"""
        cost = self.word_info[word]['cost']
        base_score = 10 / (cost + 1)
        return base_score * (1 + self.cost_adjustments.get(word, 0))

    def should_counter(self, cluster1, cluster2):
        """Reguli adaptive între clustere"""
        rules = {
            'low_cost': ['high_power', 'complex'],
            'high_power': ['neutral', 'complex'],
            'complex': ['low_cost'],
            'neutral': ['high_power']
        }
        return cluster2 in rules.get(cluster1, [])

    def dynamic_cost_adjustment(self):
        """Ajustează costurile pe baza performanței istorice"""
        for word in self.all_words:
            successes = sum(
                1 for h in self.history if h['word'] == word and h['success'])
            failures = sum(
                1 for h in self.history if h['word'] == word and not h['success'])

            if successes + failures > 0:
                success_rate = successes / (successes + failures)
                adjustment = (success_rate - 0.5) * self.learning_rate
                self.cost_adjustments[word] += adjustment
                self.cost_adjustments[word] *= self.decay_factor

    def choose_word(self, system_word):
        """Alege cel mai bun counter pentru cuvântul sistem"""
        system_word = system_word.lower()
        best_score = -np.inf
        best_choice = None

        for candidate in self.all_words:
            if candidate == system_word:
                continue

            # Calcul scor strategic
            base_score = self.relationship_matrix[system_word][candidate]
            cost = self.word_info[candidate]['cost']
            usage_penalty = np.log(self.word_info[candidate]['uses'] + 1)
            cluster_bonus = 2 if self.should_counter(
                self.category_clusters[candidate],
                self.category_clusters[system_word]
            ) else 0

            total_score = (base_score * 2 + cluster_bonus -
                           cost / 2 - usage_penalty)

            if total_score > best_score:
                best_score = total_score
                best_choice = candidate

        # Actualizează istoric
        self.word_info[best_choice]['uses'] += 1
        return best_choice

    def update_model(self, result):
        """Actualizează modelul pe baza rezultatului rundei"""
        self.history.append(result)
        system_word = result['system_word'].lower()
        used_word = result['used_word'].lower()

        # Actualizează matricea de relații
        if result['success']:
            self.relationship_matrix[system_word][used_word] += 1
            # Întărește relațiile similare
            for word in self.all_words:
                if self.category_clusters[word] == self.category_clusters[used_word]:
                    self.relationship_matrix[system_word][word] += 0.2
        else:
            self.relationship_matrix[system_word][used_word] -= 1
            # Reducere relații similare
            for word in self.all_words:
                if self.category_clusters[word] == self.category_clusters[used_word]:
                    self.relationship_matrix[system_word][word] -= 0.2

        # Ajustare dinamică a costurilor
        self.dynamic_cost_adjustment()

    def get_cost(self, word):
        """Returnează costul ajustat al cuvântului"""
        return self.word_info[word.lower()]['cost'] * (1 - self.cost_adjustments.get(word.lower(), 0))


# Exemplu de utilizare
if __name__ == "__main__":
    ai = EnhancedGameAI()

    # Simulare runde
    test_cases = [
        {'system_word': 'Lion', 'result': True},
        {'system_word': 'Virus', 'result': False},
        {'system_word': 'Earthquake', 'result': True}
    ]

    for i, case in enumerate(test_cases):
        system_word = case['system_word']
        chosen_word = ai.choose_word(system_word)
        cost = ai.get_cost(chosen_word)
        print(f"Round {i+1}:")
        print(
            f"System: {system_word} | Chosen: {chosen_word} (Cost: ${cost:.2f})")

        # Actualizează modelul cu rezultatul (simulat)
        ai.update_model({
            'system_word': system_word,
            'used_word': chosen_word,
            'success': case['result']
        })

    # Afișează învățarea finală
    print("\nMatrice de relații învățate:")
    for word in ai.all_words[:3]:
        print(f"{word}: {dict(ai.relationship_matrix[word])}")
