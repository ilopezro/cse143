# ---------------------------
# CSE 143 Assignment #1 
# 1.21.2020
# Professor Jeffrey Flanigan
#
# Isai Lopez Rodas
# ilopezro
#
# Homework Partners:
# Jennifer Dutra and ()
# ----------------------------

trainingData = open("./data/1b_benchmark.train.tokens", "r")
tokens = {}

i = 0

for line in trainingData:
	wordArray = line.split()
	wordArray.insert(0, "<start>")
	wordArray.append("<end>")
	for word in wordArray:
		if word not in tokens:
			tokens[word] = 1
		else:
			nextVal = tokens[word] + 1
			tokens[word] = nextVal
	del tokens["<start>"]

unkCounter = 0
keysToDelete = []

for key, value in tokens.items():
	if value < 3:
		unkCounter += 1
		keysToDelete.append(key)

for key in keysToDelete:
	del tokens[key]

tokens["UNK"] = unkCounter

print(len(tokens))
