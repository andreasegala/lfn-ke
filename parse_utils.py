import networkx as nx
import nltk
from graph_utils import GraphMaker
import matplotlib.pyplot as plt
import re
class Article:
    def __init__(self, docID, abstract, category,keywords) -> None:
        self.docID= docID
        cleanAbstract =re.sub("['\",]", "", abstract) 
        self.abstract = cleanAbstract
        self.category = category
        self.graph = nx.Graph()
        self.kw = keywords #che siano già da stemmare??
    
    def setGraph(self, graphmaker)-> None:
        self.graph = graphmaker.buildGraph(self.abstract)

    def printGraph(self, k, font_size):
        fig = plt.figure(figsize=(30,20))
        color_map=['blue']  
        pos = nx.spring_layout(self.graph, k)
        d = nx.degree(self.graph)
        node_sizes = []
        for i in d:
            _, value = i
            node_sizes.append(value)
        
        color_list = []

        #adjust bc they are just stems now 
        for i in self.graph.nodes:
            value = nltk.pos_tag([i])[0][1]
            if (value=='NN' or value=='NNP' or value=='NNS'):
                color_list.append('red')
            elif value=='JJ':
                color_list.append('yellow')
            else:
                color_list.append('blue')
        
        plt.figure(figsize=(40,40))
        nx.draw(self.graph, pos, node_size=[(v+1)*200 for v in node_sizes], with_labels=True, node_color=color_list, font_size=font_size)
        plt.show()


    def toString(self) -> str:
        return ''
    
    


