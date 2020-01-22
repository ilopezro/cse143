# ---------------------------
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

# Global Variables
# TOKENS holds the dictionary of all tokens in test data and 
# number of times they appear { token: numAppearances }
#
# DELETED_KEYS is a list of all UNK keys

TOKENS = {}
DELETED_KEYS = [] 

# getTrainingData() will prefill TOKENS and DELETED_KEYS to be 
# used in Dev data. 

def getTrainingData():
	trainingData = open("./data/1b_benchmark.train.tokens", "r")

	for line in trainingData:
		wordArray = line.split()
		wordArray.insert(0, "<start>")
		wordArray.append("<end>")
		for word in wordArray:
			if word not in TOKENS:
				TOKENS[word] = 1
			else:
				nextVal = TOKENS[word] + 1
				TOKENS[word] = nextVal
		del TOKENS["<start>"]

	unkCounter = 0

	for key, value in TOKENS.items():
		if value < 3:
			unkCounter += TOKENS[key]
			DELETED_KEYS.append(key)

	for key in DELETED_KEYS:
		del TOKENS[key]

	TOKENS["UNK"] = unkCounter
	trainingData.close()

# getDevData() will deal with Dev tokens and will parse through to 
# replace all UNK tokens  

def getDevData():
	devData = open("./data/1b_benchmark.dev.tokens", "r")

	for line in devData:
		wordArray = line.split()
		wordArray.insert(0, "<start>")
		wordArray.append("<end>")

		for word in wordArray:
			for key in DELETED_KEYS:
				if word == key:
					index = wordArray.index(key)
					wordArray[index] = "UNK"
	devData.close()


def main():
	print("in main")
	getTrainingData()
	getDevData()

if __name__ == "__main__":
	main()
