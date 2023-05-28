import copy
import nltk
from nltk import PorterStemmer
import re
import pickle
from collections import defaultdict
import time
import heapq
from math import log10


class SearchEngine():
    def __init__(self, name = "inverted_index.txt"):
        self.invIndex = open(name, 'r', encoding='UTF8')
        self.totalNumDocs = 0    
        self.indexGuide = dict() 
        self.urlMap = dict() 
        self.docInfo = []       # stores token and its postings from the inverted index
        self.queryInfo = dict() # stores query words and their idf scores
        self.champion = []      # stores matched document IDs up to 10

    def __del__(self):
        self.invIndex.close()
    
    def clear(self):
        self.docInfo = []   
        self.queryInfo = dict()
        self.champion = []

    # find the DocIDs that contain all the input query words
    def findCommon(self, queryList):
        toBeMerged = [] 
        result = []             
        for token, plist in self.docInfo: 
            if token in [x[0] for x in queryList]:
                toBeMerged.append(plist)             

        while True:
            largestDocID = 0
            noMoreCommon = False 
            # find largest docID within the first index to use later
            for pList in toBeMerged: 
                if pList != []:
                    if largestDocID == 0 or pList[0][0] > largestDocID:
                        largestDocID = pList[0][0]
                        largestPosting = pList[0]
                else:
                    noMoreCommon = True # if empty exists, there will be no more common docIDs
            if noMoreCommon:
                break  
        
            allCommon = True  # flag that shows if every value in the same index are same
            total_idf = 0     # total tf-idf score for all common DocIDs
            for pList in toBeMerged:
                if pList:
                    currentDocID = pList[0][0]
                    if currentDocID < largestDocID: 
                        # consider one of the DocIDs doesn't match, which means not all in common
                        allCommon  = False 
                        pList.pop(0)               
                    elif currentDocID == largestDocID:
                        total_idf += pList[0][1]
            if allCommon:
                result.append([largestDocID, total_idf]) 
                for pList in toBeMerged:
                    pList.pop(0)     
        
        # sort the DocIDs by their td-idf scores and return a list of them
        return [x[0] for x in sorted(result, key = lambda x: x[1], reverse = True)]


    def rank(self):
        cases = [] # stores possible combinations of indexes for N-1 merge
        queryInfoLength = len(self.queryInfo)

         # CASE: more than 2 query words
        if queryInfoLength > 2:
            # create possible cases for each N-1 in the order of word importance (1st word has the highest idf)
            if queryInfoLength == 4:
                cases = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]] 
            elif queryInfoLength == 3:
                cases = [[0,1], [0,2], [1,2]] 

            # find the DocIds that has all the query words in decreasing order of idf
            self.champion.extend(self.findCommon(self.queryInfo))
            if len(self.champion) >= 10:
                # stop ranking when 10 best docIDs are already found
                return           
                
            # if not enough number of results, check remaining possiblities, which is defined above
            temp = []
            for case in cases:
                for i in case:
                    temp.append(self.queryInfo[i])
                self.champion.extend(self.findCommon(temp))
                if len(self.champion) >= 10:
                    # stop ranking when 10 best docIDs are already found     
                    return
            
            # perform search on the highest idf query when there aren't any common DocIDs for N and N-1
            if len(self.champion) < 10:
                for plist in heapq.nlargest(10, self.docInfo[0][1], key=lambda x: x[1]):
                    self.champion.append(plist[0])
            

        # CASE: just 2 query words
        elif queryInfoLength == 2:
            # find the DocIDs that contain both query words 
            self.champion.extend(self.findCommon(self.queryInfo)) 
            
            # if less than 10 best DocIDs found, process ranking on the highest tf-idf word, and then the second higest if necessary
            if len(self.champion) < 10:
                best10 = []
                best10.extend(heapq.nlargest(10, self.docInfo[0][1], key=lambda x: x[1]))
                if len(best10) < 10: 
                    best10.extend(heapq.nlargest(10, self.docInfo[1][1], key=lambda x: x[1]))
                for plist in best10:
                    self.champion.append(plist[0])

    def processQuery(self, query):
        queryList = self._tokenizer(query)
        start = time.perf_counter()  # time
        validQueryList = self.get_docInfo(queryList)   
        end = time.perf_counter()   # time
        print("Find Time:", (end-start) * 1000, "ms")   # time
        
        # stop procesisng when no docInfo is retrieved from the input query words
        if len(self.docInfo) == 0:
            return

        # compute idf for query words and assign self.queryInfo
        for token, plist in self.docInfo:
            self.queryInfo[token] = log10(self.totalNumDocs/len(plist))

        # find the 10 higest tf-idf DocIDs for a single word query
        if len(validQueryList) == 1:
            best10 = heapq.nlargest(10, self.docInfo[0][1], key=lambda x: x[1]) 
            for plist in best10: 
                self.champion.append(plist[0]) 
                
        # find common docIDs and rank for more than 2 query words
        else:
            self.queryInfo = sorted(self.queryInfo.items(), key = lambda x: x[1], reverse = True) 
            self.queryInfo = self.queryInfo[:4] # focus on the 4 higest idf query words for ranking
            start = time.perf_counter() # time
            self.rank()
            end = time.perf_counter() # time
            print("Merge Time:", (end-start) * 1000, "ms") # time
   
    # seek the targeted docInfo from inverted index using indexGuide(bookkeeping file) 
    def get_docInfo(self, queryList):
        self.docInfo = []
        validQueryList = []
        for query in queryList:
            indexPosition = self._getindexPosition(query)
            if indexPosition:   
                self.invIndex.seek(indexPosition)
                line = eval(self.invIndex.readline())
                self.docInfo.append(line)
                validQueryList.append(query)
        return validQueryList

    def _getindexPosition(self, token):
        try:
            return self.indexGuide[token] 
        except:
            return False
 
    def loadIndexingData(self):
        with open('cache/indexGuide.pkl', 'rb') as i:
            self.indexGuide = pickle.load(i)
        with open('cache/urlMap.pkl', 'rb') as u:
            self.urlMap = pickle.load(u)
        with open('cache/totalNumDocs.pkl', 'rb') as df:
            self.totalNumDocs = pickle.load(df)
    
    def displayResult(self):
        matchedNum = len(self.champion)
        result = "\n"
        if matchedNum == 0:
            print("No matches found for your query.")
            return
        elif matchedNum < 5:
            result += "Displaying "+ str(matchedNum) +" matches found:\n"
        else: 
            result += "Displaying the top 10 URLs of " + str(matchedNum) + " mathches:\n"
        for docID in self.champion[:10]:
            result += self.urlMap[docID] + "\n"
        print(result)

    def _tokenizer(self, query):
        tokens = []
        pattern = re.compile("[^a-zA-Z0-9]")
        for token in nltk.word_tokenize(query):
            token = token.lower().strip()
            if not re.search(pattern, token):
                tokens.append(token)
        stemmed_tokens = [PorterStemmer().stem(token) for token in tokens] 
        return stemmed_tokens

if __name__ == "__main__":
    se = SearchEngine() 
    se.loadIndexingData() # load pickle data 
    
    print("\nWelcome to Team Kakao's Search Engine!")
    while True:
        try:
            query = input("Enter your query (type 'quit' to exit): ")
            if query.strip() == "":
                raise EOFError
            if query.lower() == "quit":
                break

            start = time.perf_counter() # time

            se.processQuery(query)
            se.displayResult()

            end = time.perf_counter()   # time
            print("Elapsed Time:", (end - start) * 1000, "ms") # time
            
            se.clear() # clearing the class variables to get the next query

        except:
            print('Please enter a valid query...')
            continue
        break
