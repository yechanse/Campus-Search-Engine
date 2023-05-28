import os
import psutil
import json
import nltk
import re
import pickle
import gc
import time
from math import log10
from nltk.stem import PorterStemmer
from collections import defaultdict
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

class Posting():
    def __init__(self, docID, termFreq = 0, weight = 0):
        self.next = None
        self.docID = docID
        self.termFreq = termFreq
        self.tagWeight = weight 

    def __str__(self):
        return f"[{self.docID}, {self.termFreq}, {self.tagWeight}]"

class PostingsList():
    def __init__(self):
        self.frequency = 0
        self.first = None
        self.last = None
    
    def __str__(self):
        repr_str = "["
        p = self.first
        while p != None:
            repr_str += f"{str(p)}, "
            p = p.next
        repr_str = repr_str[:-2]
        repr_str += "]"
        return repr_str

    def append(self, p):
        if self.first == None:
            self.first = p
            self.last = p
        else:
            self.last.next = p
            self.last = p
    
class InvIndex():
    def __init__(self):
        self.invIndex = defaultdict(PostingsList) # temporary in-memory inverted index 
        self.urlMap = dict()                      # mapping docID and URL
        self.numOfPartialFiles = 0

    def update(self, filePath, docID):
        url, soup = self._read_json(filePath)     # retrieve URL and soup content from json
        tokens = self._tokenizer(soup)            # tokenize and perform Porterstemming 
        self._updateInvIndex(tokens, docID)       # update tokens and docID to inverted index
        self.urlMap[docID] = url                  # update docID and its URL to urlMap

    def _updateInvIndex(self, tokens, docID):
        parsedTokenInfo = defaultdict(lambda: [0, 0.0]) # {token: (tf, tagWeight)} 
        for token, tag in tokens:                       # tokens = [(token, tag), .....]
            parsedTokenInfo[token][0] += 1              # count tf
            w = self._tagWeight(tag)
            if parsedTokenInfo[token][1] < w:
                parsedTokenInfo[token][1] = w

        for token, info in parsedTokenInfo.items(): 
            self.invIndex[token].frequency += 1         # count df
            tf = 1 + log10(float(info[0]))
            self.invIndex[token].append(Posting(docID, tf, info[1]))

    # assign html tags to some weights based on their importance
    def _tagWeight(self, tag):
        weight = {'title': 1.5, 'h1': 1.5, 'h2': 1.3, 'h3': 1.2, 'b': 1.2, 'strong': 1.2, 'a': 1}     
        if tag not in weight:
            return 1    
        return weight[tag]
    
    def clear(self): ##################################
        del self.invIndex
        gc.collect()
        self.invIndex = defaultdict(PostingsList)

    def dump(self):
        with open(f"cache/Partial_{self.numOfPartialFiles}.txt", "w", encoding='UTF8') as f:
            for token, docInfo in sorted(self.invIndex.items()):
                line = "[" + f"'{token}', [{str(docInfo.frequency)}, {str(docInfo)}]" + "]\n"
                f.write(line)
        self.numOfPartialFiles += 1
            
    def merge(self, totalCounter):
        temp = 0
        indexGuide = defaultdict(int)
        if os.path.exists('inverted_index.txt'): 
            os.remove('inverted_index.txt')

        mergedFile = open(f"inverted_index.txt", 'a', encoding = 'UTF8')
        FILE_OBJECTS = [open(f"cache/Partial_{idx}.txt", "r", encoding='UTF8') for idx in range(0, self.numOfPartialFiles)]
        command = [1 for x in range(0, self.numOfPartialFiles)]
        ENDING_command = [2 for x in range(0, self.numOfPartialFiles)]
        currentLine = [None for x in range(0, self.numOfPartialFiles)]
        
        while command != ENDING_command:
            for idx, io_object in enumerate(FILE_OBJECTS):
                if command[idx] == 1:
                    line = io_object.readline()
                    if line:
                        currentLine[idx] = eval(line)
                    else:
                        command[idx] = 2
                        currentLine[idx] = None
            if command == ENDING_command:
                break

            priority = None
            for d in currentLine:
                if d:
                    token = d[0]
                    if token:
                        if priority == None:
                            priority = token
                        elif token < priority:
                            priority = token
            
            winners = []
            for idx, d in enumerate(currentLine):
                if d:
                    token = d[0]
                    if token:
                        if token == priority:
                            command[idx] = 1
                            winners.append(idx)
                        else:
                            command[idx] = 0

            result = None
            for idx in winners:
                if result == None:
                    result = currentLine[idx] 
                    result = [result[0], result[1][1]]
                else:
                    result[1].extend(currentLine[idx][1][1])

                raw_posting_score = []

                for id, tf, weight in result[1]:
                    # it is idf score for each document
                    score = tf * log10(totalCounter / len(result[1])) * weight
                    raw_posting_score.append([id, score]) # 나중에 안되면 자르는걸로
            result[1] = raw_posting_score

            mergedFile.write(str(result)+'\n')

            
            i = result[0]
            if i not in indexGuide.keys():
                indexGuide[i] = temp
            temp += len(str(result)) + 2   # add 2 to explore the next line for seek()

        mergedFile.close()
        return indexGuide

    def _tokenizer(self, soup):
        tokens = [] 
        pattern = re.compile("[^a-zA-Z0-9]")
        tags = soup.find_all(['title', 'h1', 'h2', 'h3', 'b', "strong", "a"], text=True) 
        for tag in tags:
            tag_text = tag.get_text().strip()
            for token in nltk.word_tokenize(tag_text):
                token = token.lower().strip()
                if not re.search(pattern, token):
                    tokens.append(tuple((token, tag.name)))
        stemmed_tokens = [(PorterStemmer().stem(token), tag) for (token, tag) in tokens] 
        return stemmed_tokens 

    def _read_json(self, filePath):  
        with open(filePath, 'r', encoding='UTF8') as file:        
            loaded = json.loads(file.read())                       
            url, content = loaded['url'], loaded['content']  
            soup = BeautifulSoup(content, features="lxml")   
            return (url, soup)

def _getPaths(filePath = 'DEV/'):
    paths = list()
    for root, dirs, files in os.walk(filePath):
        for name in files:
            if name.endswith('.json'):
                paths.append(os.path.join(root, name))
    return paths

def pickleDump(variable, filePath):
    with open(filePath, 'wb') as f:
        pickle.dump(variable, f)

if __name__ == "__main__":
    counterLimit = 10000

    idx = InvIndex()
    targetPaths = _getPaths('DEV/')
    uniqueDocID = 1 
    jsonCounter = 0

    flag = False
    while targetPaths != []:
        idx.update(targetPaths.pop(0), uniqueDocID)
        uniqueDocID += 1
        jsonCounter += 1
        flag = True
        if jsonCounter >= counterLimit: 
            idx.dump()
            idx.clear()
            jsonCounter = 0
            flag = False

    if flag:
        idx.dump()
        idx.clear()

    indexGuide = idx.merge(uniqueDocID) 
    
    pickleDump(indexGuide, "cache/indexGuide.pkl")
    pickleDump(idx.urlMap, "cache/urlMap.pkl")
    pickleDump(int(uniqueDocID), "cache/totalNumDocs.pkl")
    