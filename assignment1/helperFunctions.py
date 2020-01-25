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
	try:
		probabilityOfBigram = probabilites[bigram]
		if bigram[0] == "<start>":
			probabilityOfHistory = 1
		else: 
			probabilityOfHistory = helperProbabilities[bigram[0]]
		return float(probabilityOfBigram)/float(probabilityOfHistory)
	except:
		return 0

# --------------------------------------------------------------------
# getTrigramPerplexity() calculates perplexity for trigrams and 
# returns the final perplexity
# --------------------------------------------------------------------
def getTrigramPerplexity(data, probabilites, helperProbabilities):
	runningSum = 0
	biggerRunningSum = 0
	sentenceLength = 0
	for sentence in data:
		sentenceArray = sentence.split()
		ngram = [(sentenceArray[i], sentenceArray[i+1], sentenceArray[i+2]) for i in range(len(sentenceArray)-2)]
		for gram in ngram:
			runningSum += math.log(getTrigramProbability(gram, probabilites, helperProbabilities), 2)
		biggerRunningSum += runningSum
		runningSum = 0
		sentenceLength += len(sentenceArray) - 1
	inverse = float(-1) / float(sentenceLength)
	exponent = inverse * biggerRunningSum; 
	return math.pow(2, exponent)

# --------------------------------------------------------------------
# getTrigramProbability() gets trigram probability
# --------------------------------------------------------------------
def getTrigramProbability(trigram, probabilites, helperProbabilities):
	try: 
		probabilityOfTrigram = probabilites[trigram]
		if trigram[0] == "<start>":
			probabilityOfHistory = 1
		else: 
			probabilityOfHistory = helperProbabilities[trigram[0:2]]
		return float(probabilityOfTrigram)/float(probabilityOfHistory)
	except:
		return 0
			

# --------------------------------------------------------------------
# getTrigramSmoothing()) gets trigram perplexity using smoothing
# --------------------------------------------------------------------
def getTrigramSmoothing(data, probabilities, helperProbabilities, unigramCount, lambdas):
	runningSum = 0
	biggerRunningSum = 0
	sentenceLength = 0
	for sentence in data:
		sentenceArray = sentence.split()
		ngram = [(sentenceArray[i], sentenceArray[i+1], sentenceArray[i+2]) for i in range(len(sentenceArray)-2)]
		for gram in ngram:
			runningMiniSum = 0
			runningMiniSum += lambdas[2] * (getTrigramProbability(gram, probabilities, helperProbabilities))
			runningMiniSum += lambdas[1] * (getBigramProbability(gram[0:2], helperProbabilities, unigramCount))
			if gram[0] == "<start>":
				numerator = 1
			else:
				numerator = unigramCount[gram[0]]
			runningMiniSum += lambdas[0] * float(numerator/sum(unigramCount.values()))
			runningSum += math.log(runningMiniSum, 2)
		biggerRunningSum += runningSum
		runningSum = 0
		sentenceLength += len(sentenceArray) - 1
	inverse = float(-1) / float(sentenceLength)
	exponent = inverse * biggerRunningSum; 
	return math.pow(2, exponent)

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

