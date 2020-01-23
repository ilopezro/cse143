import sys

TRAINING_TOKENS = {}
DELETED_KEYS = []
DEV_DATA_ARRAY = []
N_GRAMS = {}

def trainData():
    trainingData = open("./data/1b_benchmark.train.tokens", "r")
    
    for line in trainingData:
        wordArray = line.split()
        wordArray.append("<end>")
        for word in wordArray:
            if word not in TRAINING_TOKENS:
                TRAINING_TOKENS[word] = 1
            else:
                TRAINING_TOKENS[word] += 1
    
    unkCounter = 0

    for key, value in TRAINING_TOKENS.items():
        if value < 3:
            unkCounter += TRAINING_TOKENS[key]
            DELETED_KEYS.append(key)
    
    for keys in DELETED_KEYS:
        del TRAINING_TOKENS[keys]

    TRAINING_TOKENS["UNK"] = unkCounter
    trainingData.close()

def getDevData():
    devData = open("./data/1b_benchmark.dev.tokens", "r")
    setDelKeys = set(DELETED_KEYS)
    
    for line in devData:
        wordArray = line.split()
        for word in wordArray:
            if word in setDelKeys:
                wordArray[wordArray.index(word)] = "UNK"
        wordArray.insert(0, "<start>")
        wordArray.append("<end>")
        DEV_DATA_ARRAY.append(" ".join(wordArray))
    
    devData.close()

def getUnigram():
    totalProbability = sum(TRAINING_TOKENS.values())
    for line in DEV_DATA_ARRAY:
        wordArray = line.split()
        wordArray.remove("<start>")
        for word in wordArray:
            if word in TRAINING_TOKENS:
                N_GRAMS[word] = TRAINING_TOKENS[word] / len(TRAINING_TOKENS.keys())
            else:
                if word not in N_GRAMS:
                    N_GRAMS[word] = 1 / len(TRAINING_TOKENS.keys())
                else:
                    numWords = N_GRAMS[word] * len(TRAINING_TOKENS.keys())
                    numWords += 1
                    print(numWords)
                    N_GRAMS[word] = numWords / len(TRAINING_TOKENS.keys())

def getBigram():
    print("getting bigram")

def getTrigram():
    print("getting trigram")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: python3 assg1.py model [-s]')
        print("   model: unigram, bigram, or trigram")
        print("      -s: optional flag that turns on smoothing")
        exit(1)

    if sys.argv[1] != "unigram" and sys.argv[1] != "bigram" and sys.argv[1] != "trigram":
        print('Usage: python3 assg1.py model [-s]')
        print("   model: unigram, bigram, or trigram")
        print("      -s: optional flag that turns on smoothing")
        exit(1)
    
    trainData()
    getDevData()

    if sys.argv[1] == "unigram":
        getUnigram()
    elif sys.argv[1] == "bigram":
        getBigram()
    else:
        getTrigram()

if __name__ == "__main__":
    main()