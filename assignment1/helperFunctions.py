# --------------------------------------------------------------------
# getUnigrams() will prefill UNIGRAMS and DELETED_KEYS to be used in 
# Dev data. 
# --------------------------------------------------------------------
def getUnigrams(count, data, delKeys):
	trainingFile = open("./data/1b_benchmark.train.tokens", "r")

	for sentence in trainingFile:
		sentenceArray = sentence.split()
		sentenceArray.append("<end>")
		for word in sentenceArray:
			if word not in count:
				count[word] = 1
			else:
				count[word] += 1
		data.append(" ".join(sentenceArray))
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
def getPerplexity(data, probabilities):
	runningSum = 0
	biggerRunningSum = 0
	wordCount = 0
	for sentence in data:
		sentenceArray = sentence.split()
		for word in sentenceArray:
			wordCount += 1
			runningSum += math.log(probabilities[word], 2)
		biggerRunningSum += runningSum
		runningSum = 0
	inverse = float(-1) / float(wordCount)
	exponent = inverse * biggerRunningSum; 
	return math.pow(2, exponent)

# --------------------------------------------------------------------
# getDevData() replaces every UNK token and adds it to DEV_DATA_ARRAY
# --------------------------------------------------------------------
def getDevData(count, array):
	devData = open("./data/1b_benchmark.dev.tokens", "r")
	for sentence in devData:
		sentenceArray = sentence.split()
		for word in sentenceArray:
			if word not in count:
				sentenceArray[sentenceArray.index(word)] = "UNK"
		sentenceArray.append("<end>")
		array.append(" ".join(sentenceArray))
	devData.close()

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