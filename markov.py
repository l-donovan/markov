import numpy as np
import random

# Configuration

FILENAME = 'frankenstein.txt'
TEXT_CHARS = 'abcdefghijklmnopqrstuvwxyz'
DROP_CHARS = '\'"'
ESC_CHAR = '\0'
DIMENSIONS = 4
WORDS = 1000

valid_chars = TEXT_CHARS + DROP_CHARS
width = len(TEXT_CHARS) + 2
size = width ** DIMENSIONS
prob_mat = np.zeros((width,) * (DIMENSIONS + 1))

def rand_weighted(weights):
    cumul_weights = np.cumsum(weights)

    r = random.random()

    for i, weight in enumerate(cumul_weights):
        if (r < weight):
            return i

    return None

def generate_word(prob_mat):
    prev_char_indexes = [0] * DIMENSIONS
    word = ''

    while True:
        weights = prob_mat[tuple(prev_char_indexes)]
        char_idx = rand_weighted(weights)
        if (char_idx == 1):
            break
        word += TEXT_CHARS[char_idx - 2]

        prev_char_indexes.append(char_idx)
        prev_char_indexes.pop(0)

    return word

# Preprocessing phase

with open(FILENAME, 'r') as f:
    text = f.read().lower()

raw_words = ' '.join(text.split('\n')).split()
valid_words = [word + ESC_CHAR for word in raw_words if all([char in valid_chars for char in word])]

# Weighting phase

for word in valid_words:
    prev_char_indexes = [0] * DIMENSIONS

    for i in range(len(word)):
        if word[i] in DROP_CHARS:
            continue

        char_idx = TEXT_CHARS.find(word[i]) + 2
        prob_mat[tuple(prev_char_indexes)][char_idx] += 1

        prev_char_indexes.append(char_idx)
        prev_char_indexes.pop(0)

# Normalization phase

idx = [-1] + [0] * (DIMENSIONS - 1)

for c in range(width ** DIMENSIONS):
    idx[0] += 1
    i = 0

    while (idx[i] >= width and i < (DIMENSIONS - 1)):
        idx[i] = 0
        i += 1
        idx[i] += 1

    mag = prob_mat[tuple(idx)].sum()

    if (mag > 0):
        prob_mat[tuple(idx)] /= mag

# Generation phase

print(f'Generating {WORDS:,} words using a {DIMENSIONS:,}-dimensional Markov chain trained on "{FILENAME}"')
print(f'Our probability matrix contains {size:,} elements')
print()

gen_words = ' '.join(generate_word(prob_mat) for i in range(WORDS))

print(gen_words)
