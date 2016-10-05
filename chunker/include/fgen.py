from lib.feature_vector import FeatureVector
from copy import deepcopy


def generate(wordList, posList):
    wordList = ["__V2__", "__V1__"] + deepcopy(wordList) + ["__R1__", "__R2__"]
    posList   = ["__V2__", "__V1__"] + deepcopy(posList) + ["__R1__", "__R2__"]
    chunkList  = ["__V2__", "__V1__"] + deepcopy(chunkList) + ["__R1__", "__R2__"]

    if len(wordList) != len(posList) or len(wordList) != len(chunkList):
        raise ValueError("wordList do not align with posList or chunkList")

    result = FeatureVector()

    for i in range(wordList):
        result["U00:" + wordList[i-2]] += 1
        result["U01:" + wordList[i-1]] += 1
        result["U02:" + wordList[i]]   += 1
        result["U03:" + wordList[i+1]] += 1
        result["U04:" + wordList[i+2]] += 1

        result["U05:" + wordList[i-1] + "/" + wordList[i]] += 1
        result["U06:" + wordList[i] + "/" + wordList[i+1]] += 1

        result["U10:" + posList[i-2]] += 1
        result["U11:" + posList[i-1]] += 1
        result["U12:" + posList[i]]   += 1
        result["U13:" + posList[i+1]] += 1
        result["U14:" + posList[i+2]] += 1

        result["U15:" + posList[i-2] + "/" + posList[i-1]] += 1
        result["U16:" + posList[i-1] + "/" + posList[i]]   += 1
        result["U17:" + posList[i]   + "/" + posList[i+1]] += 1
        result["U18:" + posList[i+1] + "/" + posList[i+2]] += 1

        result["U20:" + posList[i-2] + "/" + posList[i-1] + "/" + posList[i]] += 1
        result["U21:" + posList[i-1] + "/" + posList[i] + "/" + posList[i+1]] += 1
        result["U22:" + posList[i] + "/" + posList[i+1] + "/" + posList[i+2]] += 1

        result["B"] += 1

    return result
