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

from helperFunctions import getUnigrams, populateUnk, getPerplexity, getProbability, getDevData

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

# --------------------------------------------------------------------
# getBigrams() prepares gets all bigrams available from trainingData
# --------------------------------------------------------------------
def getBigrams(data):
	print("in here")

if __name__ == "__main__":
	unigramCount = {}
	unigramProbabilities = {}
	bigramCount = {}
	bigramProbabilities = {}
	deletedKeys = []
	trainingData = []
	devData = []

	print("-------------------------------------------------------------")
	print("Unigrams")
	print("-------------------------------------------------------------")
	print("Getting Unigram Data Set Up")
	getUnigrams(unigramCount, trainingData, deletedKeys)
	print("Populating Training Data with UNKs")
	populateUnk(trainingData, unigramCount)
	print("Calculating Probabilities for all Unigram Tokens")
	getProbability(unigramCount, unigramProbabilities)
	print("Calculating Perplexity for Training Data")
	print(f"Perplexity of Training Data for Unigram is: {getPerplexity(trainingData, unigramProbabilities)}\n")

	print("Getting Dev Data Set Up")
	getDevData(unigramCount, devData)
	print("Calculating Perplexity for Dev Data")
	print(f"Perplexity of Dev Data for Unigram is: {getPerplexity(devData, unigramProbabilities)}")
	print("-------------------------------------------------------------")
	print("Bigrams")
	print("-------------------------------------------------------------")
	print("Getting Bigrams Data Set Up")
	print("Adding <start> Token to trainingData")
	addStartToken(trainingData)
	print("Getting Dev Data Set Up")
	print("Adding start token to devData")
	addStartToken(devData)

