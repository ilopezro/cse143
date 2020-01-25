import math

# --------------------------------------------------------------------
# getUnigrams() will prefill UNIGRAMS and DELETED_KEYS to be used in 
# Dev data. 
# --------------------------------------------------------------------
def getUnigrams(count, delKeys):
	trainingFile = open("./data/1b_benchmark.train.tokens", "r")

	for sentence in trainingFile:
		sentenceArray = sentence.split()
		sentenceArray.append("<end>")
		for word in sentenceArray:
			if word not in count:
				count[word] = 1
			else:
				count[word] += 1
	trainingFile.close()

	count["UNK"] = 0

	for key, value in count.items():
		if value < 3:
			delKeys.append(key)
			count["UNK"] += value
	
	for key in delKeys:
		del count[key]

	trainingFile.close()

# --------------------------------------------------------------------
# populateUnks() populats unknown tokens with UNKs in the given 
# sentence list
# --------------------------------------------------------------------
def populateUnk(data, model):
	for sentence in data:
		index = data.index(sentence)
		sentenceArray = sentence.split()
		for word in sentenceArray:
			if word not in model:
				sentenceArray[sentenceArray.index(word)] = "UNK"
		data[index] = " ".join(sentenceArray)

# --------------------------------------------------------------------
# getUnigramProbability() populates UNIGRAM_PROBABILITY with all the 
# probabilites within Unigram 
# --------------------------------------------------------------------
def getProbability(count, probabilites):
	for key, value in count.items():
		probabilites[key] = value/sum(count.values())

# --------------------------------------------------------------------
# getUnigramPerplexity() calculates perplexity for unigrams and 
# returns the final perplexity
# --------------------------------------------------------------------
def getUnigramPerplexity(data, probabilities):
	runningSum = 0
	biggerRunningSum = 0
	wordCount = 0
	for sentence in data:
		sentenceArray = sentence.split()
		for word in sentenceArray:
			if word == "<start>":
				continue
			runningSum += math.log(probabilities[word], 2)
			wordCount += 1
		biggerRunningSum += runningSum
		runningSum = 0
	inverse = float(-1) / float(wordCount)
	exponent = inverse * biggerRunningSum; 
	return math.pow(2, exponent)

# --------------------------------------------------------------------
# getBigramPerplexity() calculates perplexity for unigrams and 
# returns the final perplexity
# --------------------------------------------------------------------
def getBigramPerplexity(data, probabilites, helperProbabilities):
	runningSum = 0
	biggerRunningSum = 0
	sentenceLength = 0
	for sentence in data:
		sentenceArray = sentence.split()
		ngram = [(sentenceArray[i], sentenceArray[i+1]) for i in range(len(sentenceArray)-1)]
		for gram in ngram:
			runningSum += math.log(getBigramProbability(gram, probabilites, helperProbabilities), 2)
		biggerRunningSum += runningSum
		runningSum = 0
		sentenceLength += len(sentenceArray) - 1
	inverse = float(-1) / float(sentenceLength)
	exponent = inverse * biggerRunningSum; 
	return math.pow(2, exponent)
			

# --------------------------------------------------------------------
# getBigramProbability() gets bigram probability
# --------------------------------------------------------------------
def getBigramProbability(bigram, probabilites, helperProbabilities):
	probabilityOfBigram = probabilites[bigram]
	if bigram[0] == "<start>":
		probabilityOfHistory = 1
	else: 
		probabilityOfHistory = helperProbabilities[bigram[1]]
	return float(probabilityOfBigram)/float(probabilityOfHistory)

# --------------------------------------------------------------------
# getTrigramProbability() gets bigram probability
# --------------------------------------------------------------------
def getTrigramProbability(bigram, probabilites, helperProbabilities):
	print("in trigram probabiloty")

# --------------------------------------------------------------------
# getData() replaces every UNK token and adds it to DEV_DATA_ARRAY
# --------------------------------------------------------------------
def getData(count, array, type):
	if type == "dev":
		data = open("./data/1b_benchmark.dev.tokens", "r")
	elif type == "test":
		data = open("./data/1b_benchmark.test.tokens", "r")
	else:
		data = open("./data/1b_benchmark.train.tokens")

	for sentence in data:
		sentenceArray = sentence.split()
		for word in sentenceArray:
			if word not in count:
				sentenceArray[sentenceArray.index(word)] = "UNK"
		sentenceArray.insert(0, "<start>")
		sentenceArray.append("<end>")
		array.append(" ".join(sentenceArray))
	data.close()

# --------------------------------------------------------------------
# addStartToken() prepares DATA_ARRAY to be used by Bigram and 
# Trigram models by adding the start token
# --------------------------------------------------------------------
def addStartToken(data):
	for sentence in data:
		index = data.index(sentence)
		sentenceArray = sentence.split()
		sentenceArray.insert(0, "<start>")
		data[index] = " ".join(sentenceArray)

# --------------------------------------------------------------------
# getNgrams() gets all n-grams available from data num >= 2
# --------------------------------------------------------------------
def getNgrams(data, count, num):
	for sentence in data:
		sentenceArray = sentence.split()
		if num == 2:
			ngram = [(sentenceArray[i], sentenceArray[i+1]) for i in range(len(sentenceArray)-1)]
		if num == 3:
			ngram = [(sentenceArray[i], sentenceArray[i+1], sentenceArray[i+2]) for i in range(len(sentenceArray)-2)]
		for gram in ngram:
			if gram not in count:
				count[gram] = 1
			else:
				count[gram] += 1
