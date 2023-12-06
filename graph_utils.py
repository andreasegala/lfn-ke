import networkx as nx
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

# A tool that generates cooccurence graphs from a text while keeping the stopwords list and the chosen stemmer
class GraphMaker:
    #Constructor that sets the stopwords and the stemmer
    def __init__(self, stopPath, stemmer):
        f =open(stopPath,'r')
        parsedSw = f.readlines()
        f.close()
        swList =[]
        for sw in parsedSw:
            swList.append(sw.rstrip())
        self.stopwords = set(swList)
        if (stemmer=='POR'):
            self.stemmer = PorterStemmer() #to change into enum or class of vals
        else:
            self.stemmer = SnowballStemmer('english')

    #routine that builds the graph    
    def buildGraph(self, text) -> nx.Graph:
        k=0.5
        
        font_size=26
        words_in_text = []
        is_noun = lambda word, pos: pos[:2] == 'NN' and (word not in self.stopwords)
        is_adjective_or_verb = lambda word , pos: (pos[:2]=='JJ' or pos[:2]=='VB') and (word not in self.stopwords)

        for sent in text.split('.')[:-1]:
            tokenized = nltk.word_tokenize(sent)
            nouns_adj_ver=[word for (word, pos) in nltk.pos_tag(tokenized) if (is_noun(word,pos) or is_adjective_or_verb(word,pos))]
            words_in_text.append(' '.join([word for word in nouns_adj_ver if not (len(word)<=2)])) #nouns_in_text becomes a list of Strings, each of which acts as list of nouns

        print(words_in_text)
        words_list = [] #will determine the numbers and the id of the rows

        for sent in words_in_text: #deleting duplicates 
            temp = sent.split(' ')
            for word in temp: #stemming and adding stems
                if self.stemmer.stem(word) not in words_list:
                    words_list.append(self.stemmer.stem(word))

        df = pd.DataFrame(np.zeros(shape=(len(words_list),2)), columns=['Stems', 'Neighbours'])
        df['Stems'] = words_list
        
        '''for sent in text.split('.'):
            for noun in nouns_list:
                if noun in sent:
                    tokenized = nltk.word_tokenize(sent)
                    adjectives_or_verbs = [word for (word, pos) in nltk.pos_tag(tokenized) if (is_adjective_or_verb(word,pos))]
                    ind = df[df['Nouns']==noun].index[0]
                    df['Verbs & Adjectives'][ind]=adjectives_or_verbs'''
        
        for sent in text.split('.'):
            tokens = nltk.word_tokenize(sent)
            stemmed_sent = [self.stemmer.stem(word) for (word, pos) in nltk.pos_tag(tokens) if is_noun(word,pos) or is_adjective_or_verb(word,pos) ]
            #print(stemmed_sent)
            for stem in words_list:
                if stem in stemmed_sent:
                    ind = df[df['Stems']==stem].index[0]
                    #temp = df.at[ind, 'Neighbours']
                    #df.at[ind, 'Neighbours']=temp.append([ss for (ss)in stemmed_sent if not(ss==stem)])
                    df['Neighbours'][ind]=[ss for ss in stemmed_sent if not(ss==stem)]

        fig = plt.figure(figsize=(30,20))
        G = nx.Graph()
        color_map=[]
        for i in range(len(df)):
            G.add_node(df['Stems'][i])
            color_map.append('blue')
            for word in df['Neighbours'][i]:
                G.add_edges_from([(df['Stems'][i], word)])

        pos = nx.spring_layout(G, k)

        d = nx.degree(G)
        node_sizes = []
        for i in d:
            _, value = i
            node_sizes.append(value)
        
        color_list = []
        for i in G.nodes:
            value = nltk.pos_tag([i])[0][1]
            if (value=='NN' or value=='NNP' or value=='NNS'):
                color_list.append('red')
            elif value=='JJ':
                color_list.append('yellow')
            else:
                color_list.append('blue')
        
        plt.figure(figsize=(40,40))
        nx.draw(G, pos, node_size=[(v+1)*200 for v in node_sizes], with_labels=True, node_color=color_list, font_size=font_size)
        plt.show()

        return G   

        
    
    


        