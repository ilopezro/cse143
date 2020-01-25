# ----------------------------
# CSE 143 Assignment #1 
# 1.21.2020
# Professor Jeffrey Flanigan
#
# Isai Lopez Rodas
# ilopezro
#
# Homework Partners:
# Jennifer Dutra and Khang Tran
# ----------------------------

import sys, math

# ----------------------------
# Global Variables
#
# UNIGRAMS_COUNT holds the dictionary of all tokens in test data and 
# number of times they appear { token: numAppearances }
#
# UNIGRAMS_PROBABILITIES holds the probabilities of the unigrams 
#
# DELETED_KEYS is a list of all UNK keys
# 
# UNIGRAM_TRAINING_DATA holds a list of all the sentences for unigram
#
# DEV_DATA_ARRAY holds a list of all sentences from dev data
# ----------------------------

UNIGRAMS_COUNT = {}
UNIGRAMS_PROBABILITIES = {}
DELETED_KEYS = []
UNIGRAM_TRAINING_DATA = []
DEV_DATA_ARRAY = []

# ----------------------------
# getUnigrams() will prefill UNIGRAMS and DELETED_KEYS to be 
# used in Dev data. 
# ----------------------------

def getUnigrams():
	trainingFile = open("./data/1b_benchmark.train.tokens", "r")

	for sentence in trainingFile:
		sentenceArray = sentence.split()
		sentenceArray.append("<end>")
		for word in sentenceArray:
			if word not in UNIGRAMS_COUNT:
				UNIGRAMS_COUNT[word] = 1
			else:
				UNIGRAMS_COUNT[word] += 1
		UNIGRAM_TRAINING_DATA.append(" ".join(sentenceArray))
	trainingFile.close()

	UNIGRAMS_COUNT["UNK"] = 0

	for key, value in UNIGRAMS_COUNT.items():
		if value < 3:
			DELETED_KEYS.append(key)
			UNIGRAMS_COUNT["UNK"] += value
	
	for key in DELETED_KEYS:
		del UNIGRAMS_COUNT[key]

	trainingFile.close()

# ----------------------------
# populateUnks() populats unknown tokens with 
# UNKs in the given sentence list
# ----------------------------

def populateUnk(data):
	for sentence in data:
		index = data.index(sentence)
		sentenceArray = sentence.split()
		for word in sentenceArray:
			if word not in UNIGRAMS_COUNT:
				sentenceArray[sentenceArray.index(word)] = "UNK"
		data[index] = " ".join(sentenceArray)

# ----------------------------
# getUnigramProbability() populates UNIGRAM_PROBABILITY 
# with all the probabilites within Unigram 
# ----------------------------
def getUnigramProbability():
	for key, value in UNIGRAMS_COUNT.items():
		UNIGRAMS_PROBABILITIES[key] = value/sum(UNIGRAMS_COUNT.values())

def getUnigramPerplexity(data):
	runningSum = 0
	biggerRunningSum = 0
	wordCount = 0
	for sentence in data:
		sentenceArray = sentence.split()
		for word in sentenceArray:
			wordCount += 1
			runningSum += math.log(UNIGRAMS_PROBABILITIES[word], 2)
		biggerRunningSum += runningSum
		runningSum = 0
	inverse = float(-1) / float(wordCount)
	exponent = inverse * biggerRunningSum; 
	return math.pow(2, exponent)

# ----------------------------
# Starting Dev Manipulation for Unigram
# ----------------------------

def getDevData():
	devData = open("./data/1b_benchmark.dev.tokens", "r")
	for sentence in devData:
		sentenceArray = sentence.split()
		for word in sentenceArray:
			if word not in UNIGRAMS_COUNT:
				sentenceArray[sentenceArray.index(word)] = "UNK"
		sentenceArray.append("<end>")
		DEV_DATA_ARRAY.append(" ".join(sentenceArray))
	devData.close()

if __name__ == "__main__":
	print("Getting Unigrams Set Up")
	getUnigrams()
	print("Populating Training Data with UNKs")
	populateUnk(UNIGRAM_TRAINING_DATA)
	print("Calculating Probabilities for all Unigram Tokens")
	getUnigramProbability()
	print("Calculating Perplexity for Training Data")
	print(f"Perplexity of Training Data for Unigram is: {getUnigramPerplexity(UNIGRAM_TRAINING_DATA)}")

	print("Getting Dev Data Set Up")
	getDevData()
	print("Calculating Perplexity for Dev Data")
	print(f"Perplexity of Dev Data for Unigram is: {getUnigramPerplexity(DEV_DATA_ARRAY)}")