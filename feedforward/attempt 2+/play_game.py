import torch
import pandas as pd
import random
import sys
sys.path.append('feedforward/attempt 2+')
# replace the following line with which training loss to use
from train_BCE_sample import models

url = "feedforward/attempt 2+/"

df_test_words = pd.read_csv(url + 'test_words.csv', keep_default_na=False)
test_words = df_test_words['test words'].tolist()
print('imported words...')

# Game playing agent
def play_hangman(model, word, max_guesses=6):
    n = len(word)
    guesses = 0
    correct_guesses = 0
    guessed_word = ['_'] * n
    guessed_letters = []
    while correct_guesses < n:
        input = torch.zeros(1, n * 27)
        for i in range(n):
            if guessed_word[i] != '_':
                input[0, i * 27 + ord(guessed_word[i]) - ord('a')] = 1
            else:
                input[0, i * 27 + 26] = 1
        output = model(input)
        temp = [(i, output[0, i].item()) for i in range(26)]
        temp.sort(key=lambda x: x[1], reverse=True)
        guess = chr(temp[0][0] + ord('a'))
        for i in range(1, 26):
            if guess not in guessed_letters:
                break
            guess = chr(temp[i][0] + ord('a'))
        guessed_letters.append(guess)
        if guess in word:
            for i in range(n):
                if word[i] == guess:
                    guessed_word[i] = guess
                    correct_guesses += 1
        else:
            guesses += 1
    return guesses

# Evaluate the model
sampled_population = random.sample(range(len(test_words)), 1000)
correct = 0
total_guesses = 0
for i in range(1000):
    word = test_words[sampled_population[i]]
    n = len(word)
    guesses = play_hangman(models[n-1], word)
    if guesses <= 6:
        correct += 1
    total_guesses += guesses
    print(f'Game {i+1}, Word: {word}, Guesses: {guesses}')
print(f'Accuracy: {correct / 1000}')
print(f'Average number of guesses: {total_guesses / 1000}')