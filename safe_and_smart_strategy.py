import requests
# from time import sleep
import random

host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5


def what_beats(word):
    # sleep(random.randint(1, 3))
    return random.randint(17, 32)


def play_game(player_id):
    for round_id in range(1, NUM_ROUNDS+1):
        round_num = -1
        while round_num != round_id:
            response = requests.get(get_url)
            # print(response.json())
            sys_word = response.json()['word']
            round_num = response.json()['round']

            # sleep(1)

        if round_id > 1:
            status = requests.get(status_url)
            # print(status.json())

        choosen_word = what_beats(sys_word)
        # print(f"Chosen word: {choosen_word}")
        data = {"player_id": player_id,
                "word_id": choosen_word, "round_id": round_id}
        response = requests.post(post_url, json=data)
        # print(response.json())


if __name__ == "__main__":
    play_game("UcI28jovUk")
