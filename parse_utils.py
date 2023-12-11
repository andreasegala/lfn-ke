import networkx as nx
import nltk
import graph_utils
import matplotlib.pyplot as plt
import re
import random
import json
import glob
import os
class Article:
    def __init__(self, docID, abstract, categories, keywords, filename) -> None:
        self.docID= docID
        cleanAbstract =re.sub("['\",]", "", abstract) 
        self.abstract = cleanAbstract
        self.categories = categories
        self.graph = nx.Graph()
        self.filename = filename
        self.kw = keywords #che siano già da stemmare??
    
    def setGraph(self, graphmaker)-> None:
        self.graph = graphmaker.buildGraph(self.abstract)

    def printGraph(self):
        graph_utils.printGraph(self.graph)


    def toString(self) -> str:
        return ''

def parse_and_sample(path_to_dir, sample_size) -> list: #a list of obj Article?? Or is it better to store everything in a dictionary?
    
    sampled_articles = []
    random.seed(3101960)
    path_to_dir_format = path_to_dir+"\*.json"
    all_paths = glob.glob(path_to_dir_format)
    
    allowed_files =[]

    #list of forbidden files
    f2 = open(".\\resources\\forbiddenFiles.txt", "r")
    forbidden_files = f2.readlines() #va fatto rstrip
    f2.close()

    for i in range(0,len(forbidden_files)):
        forbidden_files[i] = forbidden_files[i].rstrip()

    #determining allowed files 
    for pp in all_paths:
        if os.path.basename(pp) not in forbidden_files:
            allowed_files.append(pp)
    
    del all_paths, forbidden_files
    #sampling from suitable candidates
    sampled_paths = random.choices(allowed_files,k =sample_size)
    
    for path in sampled_paths:
        #scan file and gather information to build article object 
        f= open(path,"r")
        #print(path+"\n")
        temp_dict = json.load(f)
        f.close()
        try:
            abstract = temp_dict["abstract"]
            keywords =temp_dict["metadata"]["keywords"]
            docID = temp_dict["docId"]
            categories = temp_dict["metadata"]["subjareas"]
            filename = os.path.basename(path)
            new_article = Article(docID, abstract, categories, keywords, filename)
            sampled_articles.append(new_article)

        except KeyError:
            f1 = open(".\\resources\\forbiddenFiles.txt", "a")
            f1.write(os.path.basename(path)+"\n")
            f1.close()

            print("Seems as if a forbidden file has escaped the check!!-> added to forbidden list!")

        del temp_dict 
    
    return sampled_articles

    



    
    