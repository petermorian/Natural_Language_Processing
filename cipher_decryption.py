# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python

import numpy as np
import matplotlib.pyplot as plt
import string
import random
import re
import requests
import os
import textwrap


### 1. create substitution cipher

letters1 = list(string.ascii_lowercase)    # keys of dict - returns all letters in alphabetical order
letters2 = list(string.ascii_lowercase)    # value of dict - returns all letters in alphabetical order
true_mapping = {}           # create empty dictionary
random.shuffle(letters2)    # populate dictionary with shuffled second set of letters
for k, v in zip(letters1, letters2):
  true_mapping[k] = v       # populate map with both lists (zip)




### 2. language model

M = np.ones((26, 26))                # initialize Markov matrix to store bigram probs (26 letters by 2)
pi = np.zeros(26)                    # initial state distribution to store unigram probs
def update_transition(ch1, ch2):     # a function to update the Markov matrix (ch1 = starting character & ch2 = ending character)
  i = ord(ch1) - 97                  # ord('a') = 97, ord('b') = 98, ...
  j = ord(ch2) - 97                  # converts each letter into their binary formats (for comps)
  M[i,j] += 1                        # add to matrix at cell i,j   
def update_pi(ch):                   # a function to update the initial state distribution pi
  i = ord(ch) - 97
  pi[i] += 1
def get_word_prob(word):             # get the log-probability of a word / token
  i = ord(word[0]) - 97              # do this for first letter 
  logp = np.log(pi[i])
  for ch in word[1:]:                #loop for remaining letters
    j = ord(ch) - 97
    logp += np.log(M[i, j])          # update prob
    i = j                            # update j
  return logp
def get_sequence_prob(words):        # get the prob of a sequence of words
  if type(words) == str:             # if input is a string, split into an array of tokens
    words = words.split()            # convert words into a list of strings (of each word)
  logp = 0
  for word in words:
    logp += get_word_prob(word)      #loop get_word_prob for eac word
  return logp

### create a markov model based on an English dataset
# is an edit of https://www.gutenberg.org/ebooks/2701
# (I removed the front and back matter)
# download the file
if not os.path.exists('moby_dick.txt'):
  print("Downloading moby dick...")
  r = requests.get('https://lazyprogrammer.me/course_files/moby_dick.txt')
  with open('moby_dick.txt', 'w') as f:
    f.write(r.content.decode())

regex = re.compile('[^a-zA-Z]')       # for replacing non-alpha characters
for line in open('moby_dick.txt'):    # load in words
  line = line.rstrip()                # there are blank lines in the file
  if line:
    line = regex.sub(' ', line)       # replace all non-alpha characters with space
    tokens = line.lower().split()     # split the tokens in the line and lowercase
    for token in tokens:              # update the model
      ch0 = token[0]                  # first letter
      update_pi(ch0)
      for ch1 in token[1:]:           # other letters
        update_transition(ch0, ch1)
        ch0 = ch1 
pi /= pi.sum()                        # normalize the probabilities
M /= M.sum(axis=1, keepdims=True)





###  3. encode a message
# this is a random excerpt from Project Gutenberg's
# The Adventures of Sherlock Holmes, by Arthur Conan Doyle
# https://www.gutenberg.org/ebooks/1661
original_message = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.

def encode_message(msg):            # a function to encode a message
  msg = msg.lower()                 # downcase all letters
  msg = regex.sub(' ', msg)         # replace non-alpha characters
  coded_msg = []                    # make the encoded message
  for ch in msg:
    coded_ch = ch                   # could just be a space
    if ch in true_mapping:
      coded_ch = true_mapping[ch]
    coded_msg.append(coded_ch)
  return ''.join(coded_msg)
encoded_message = encode_message(original_message)
def decode_message(msg, word_map):    # a function to decode a message
  decoded_msg = []                    # create empty message
  for ch in msg:                      # loop for each letter
    decoded_ch = ch                   # could just be a space
    if ch in word_map:                # return matching encoded letter to key
      decoded_ch = word_map[ch]       
    decoded_msg.append(decoded_ch)
  return ''.join(decoded_msg)






### 4. run an evolutionary algorithm to decode the message

dna_pool = []                            # this is our initialization point
for _ in range(20):
  dna = list(string.ascii_lowercase)
  random.shuffle(dna)
  dna_pool.append(dna)
def evolve_offspring(dna_pool, n_children):
  offspring = []                        # make n_children per offspring
  for dna in dna_pool:
    for _ in range(n_children):
      copy = dna.copy()
      j = np.random.randint(len(copy))  #randomly swap two positions
      k = np.random.randint(len(copy))
      tmp = copy[j]                     # switch
      copy[j] = copy[k]
      copy[k] = tmp         
      offspring.append(copy)        
  return offspring + dna_pool           

num_iters = 1000                              # number of loops
scores = np.zeros(num_iters)             
best_dna = None 
best_map = None
best_score = float('-inf')
for i in range(num_iters):        
  if i > 0:                                       # get offspring from the current dna pool
    dna_pool = evolve_offspring(dna_pool, 3)      # if first gen, no need for offspring
  dna2score = {}                                  # calculate score for each dna
  for dna in dna_pool:
    current_map = {}                              # populate map
    for k, v in zip(letters1, dna):
      current_map[k] = v
    decoded_message = decode_message(encoded_message, current_map)  #decode mesage
    score = get_sequence_prob(decoded_message)    #get log prob of decoded message
    dna2score[''.join(dna)] = score               # store it
    if score > best_score:                        # record the best so far
      best_dna = dna
      best_map = current_map
      best_score = score                     
  scores[i] = np.mean(list(dna2score.values()))   # average score for this generation
  sorted_dna = sorted(dna2score.items(), key=lambda x: x[1], reverse=True)  #sort dictionary of scores
  dna_pool = [list(k) for k, v in sorted_dna[:5]]  # keep the best 5 dna
  if i % 200 == 0:
    print("iter:", i, "score:", scores[i], "best so far:", best_score)
decoded_message = decode_message(encoded_message, best_map)    # use best score
print("LL of decoded message:", get_sequence_prob(decoded_message))
print("LL of true message:", get_sequence_prob(regex.sub(' ', original_message.lower())))
for true, v in true_mapping.items():             # which letters are wrong?
  pred = best_map[v]
  if true != pred:
    print("true: %s, pred: %s" % (true, pred))
print("Decoded message:\n", textwrap.fill(decoded_message)) # print the final decoded message
print("\nTrue message:\n", original_message)
plt.plot(scores)
plt.show()

# notes
# sometimes the LL of the true message maybe worse than our decoded message
# incorrect letters can be due to unlikely bigrams (e.g. dOZen vs dOKen)





