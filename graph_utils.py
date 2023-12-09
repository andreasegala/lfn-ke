import networkx as nx
import os
import nltk
import numpy as np
import pandas as pd
import random
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
        is_noun = lambda word, pos: pos[:2] == 'NN' and not (word in self.stopwords)
        is_adjective_or_verb = lambda word , pos: (pos[:2]=='JJ' or pos[:2]=='VB') and not (word in self.stopwords)

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

        df = pd.DataFrame(np.zeros(shape=(len(words_list),2)), columns=['Stems', 'neighbors'])
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
                    #temp = df.at[ind, 'neighbors']
                    #df.at[ind, 'neighbors']=temp.append([ss for (ss)in stemmed_sent if not(ss==stem)])
                    df['neighbors'][ind]=[ss for ss in stemmed_sent if not(ss==stem)]

        fig = plt.figure(figsize=(30,20))
        G = nx.Graph()
        color_map=[]
        for i in range(len(df)):
            G.add_node(df['Stems'][i])
            color_map.append('blue')
            for word in df['neighbors'][i]:
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

def localPageRankApprox(G) -> dict:
    nodes_list = G.nodes
    n = len(nodes_list)
    alpha = 0.85
    computed_centralities =  {} #dizionario per le centralità calcolate
    r = 10
    for u in nodes_list:
        #initializing for t=0
        PR_u = np.zeros(r+1)
        PR_u[0] = 1
        layers = np.empty(r+1, dtype =dict)
        layers[0] = {}
        layers[0][u] = 1
        inf_dict = np.empty(r+1, dtype= dict) #at each lvl there will be inf_t(x,u) with key x and value u 
        inf_dict[0] ={}
        inf_dict[0][u] =1
        for t in range(1,r+1):
            layers[t]={}
            inf_dict[t] ={}
            #initializing layer t: we put in layer t only neighbors of nodes in layers[t-1] che non sono già stati visitati: elif caso 0
            last_visited = layers[t-1]
            for v in last_visited:
                for w in G.neighbors(v):
                    #se w è un vicino che non è ancora stato visitato
                    if t>1:
                        if w not in last_visited and not w in layers[t-2]:
                            layers[t][w] = 1
                    else:
                        layers[t][w]=1
            #ho inizializzato il layer t
            for v in layers[t]:
                s=0
                for w in layers[t-1]:
                    if w in G.neighbors(v):
                        s += inf_dict[t-1][w]
                inf_dict[t][v] = (1/G.degree[v]) * s
            
            totSum = sum(layers[t][key] for key in layers[t])
            PR_u[t] = PR_u[t-1] + ((1-alpha)*(alpha**t)/n)* totSum
        computed_centralities[u] = PR_u[r]
    return computed_centralities

def improvedEstimateLCC(G, p) -> dict:
    edges_list = G.edges
    S =[]#sample of edges
    #sono grafi piccoli e stanno in memoria 
    t_S ={}
    deg ={}
    cc ={}
    random.seed(os.urandom)
    for v in G.nodes:
        t_S[v]=0
        deg[v] =0
    
    for edge in edges_list:
        u = edge[0]
        v = edge[1]
        deg[v] +=1
        deg[u] +=1
        
        N_S =[] #set of vertices
        N_s_v =[]
        for x in G.neighbors(v):
            if (x,v) in S:
                N_s_v.append(x)
            elif (v,x) in S:
                N_s_v.append(x)
        
        for x in G.neighbors(u):
            if (u,x) in S:
                if x in N_s_v:
                    N_S.append(x)
            elif(x,u) in S:
                if x in N_s_v:
                    N_S.append(x)
        
        for c in N_S:
            t_S[c] +=1
            t_S[u] +=1
            t_S[v] +=1

        if random.random()<p:
            S.append(edge)
    
    for v in G.nodes:
        cc[v] = (1/p**2)*2*t_S[v]/(deg[v]*(deg[v]-1))

    return cc  


    

                

                    
            

    
        


        


        
    
    


        