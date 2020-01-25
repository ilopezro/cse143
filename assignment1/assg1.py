# --------------------------------------------------------------------
# CSE 143 Assignment #1 
# 1.21.2020
# Professor Jeffrey Flanigan
#
# Isai Lopez Rodas
# ilopezro
#
# Homework Partners:
# Jennifer Dutra and Khang Tran
# --------------------------------------------------------------------

import math
from helperFunctions import getUnigrams, populateUnk, getUnigramPerplexity, getProbability, getData, addStartToken, getNgrams, getBigramPerplexity, getTrigramPerplexity

# --------------------------------------------------------------------
# Variables
#
# unigramCount holds the dictionary of all tokens in test data and 
# number of times they appear { token: numAppearances }
#
# unigramProbabilities holds the probabilities of the unigrams 
# 
# bigramCount holds the dictionary of all tokens in test data and 
# number of times they appear { token: numAppearances }
#
# bigramProbabilities holds the probabilities of the unigrams 
#
# deletedKeys is a list of all UNK keys
# 
# trainingData holds a list of all the sentences for training data
#
# devData holds a list of all sentences from dev data
# --------------------------------------------------------------------

if __name__ == "__main__":

	unigramCount = {}
	unigramProbabilities = {}
	bigramCount = {}
	trigramCount = {}
	deletedKeys = []
	trainingData = []
	devData = []
	testData = []

	print("-------------------------------------------------------------")
	print("Unigrams")
	print("-------------------------------------------------------------")
	print("Getting Unigram Data")
	getUnigrams(unigramCount, deletedKeys)
	print("Populating Training Data with UNKs")
	getData(unigramCount, trainingData, type="train")
	print("Calculating Probabilities for all Unigram Tokens")
	getProbability(unigramCount, unigramProbabilities)
	print("Calculating Perplexity for Training Data")
	print(f"Perplexity of Training Data for Unigram is: {getUnigramPerplexity(trainingData, unigramProbabilities)}\n")

	print("Getting Dev Data Set Up")
	getData(unigramCount, devData, type="dev")
	print("Calculating Perplexity for Dev Data")
	print(f"Perplexity of Dev Data for Unigram is: {getUnigramPerplexity(devData, unigramProbabilities)}\n")

	# print("Getting Test Data Set Up")
	# getData(unigramCount, testData, type="test")
	# print("Calculating Perplexity for Dev Data")
	# print(f"Perplexity of Test Data for Unigram is: {getUnigramPerplexity(testData, unigramProbabilities)}\n")
	print("-------------------------------------------------------------")
	print("Bigrams")
	print("-------------------------------------------------------------")
	print("Getting Bigrams Data")
	print("Finding Bigrams in Training Data")
	getNgrams(trainingData, bigramCount, 2)
	print("Finding Probabilities in Training Data for all Bigrams")
	print("Calculating Perplexity for Training Data")
	print(f"Perplexity of Training Data for Bigram is: {getBigramPerplexity(trainingData, bigramCount, unigramCount)}\n")

	print("Getting Dev Data")
	print("Calculating Perplexity for Dev Data")
	try:
		print(f"Perplexity of Dev Data for Bigram is: {getBigramPerplexity(devData, bigramCount, unigramCount)}\n")
	except:
		print(f"Perplexity of Dev Data for Bigram is: {math.inf}\n")
	
	# print("Getting Test Data")
	# print("Calculating Perplexity for Test Data")
	# print(f"Perplexity of Test Data for Bigram is: {getBigramPerplexity(testData, bigramCount, unigramCount)}\n")
	
	print("-------------------------------------------------------------")
	print("Trigrams")
	print("-------------------------------------------------------------")

	print("Getting Trigrams Data Set Up")
	print("Finding Trigrams in Training Data")
	getNgrams(trainingData, trigramCount, 3)

	print("Finding Probabilities in Training Data for all Trigram")
	# getProbability(trigramCount, trigramProbabilities)

	print("Calculating Perplexity for Training Data")
	print(f"Perplexity of Training Data for Trigram is: {getTrigramPerplexity(trainingData, trigramCount, bigramCount)}\n")

	print("Getting Dev Data Set Up")
	print("Calculating Perplexity for Dev Data")
	try:
		print(f"Perplexity of Dev Data for Unigram is: {getTrigramPerplexity(devData, trigramCount, bigramCount)}\n")
	except:
		print(f"Perplexity of Dev Data for Unigram is: {math.inf}\n")

	# print("Getting Test Data")
	# print("Calculating Perplexity for Test Data")
	# print(f"Perplexity of Test Data for Trigram is: {getPerplexity(testData, trigramCount, bigramCount)}\n")