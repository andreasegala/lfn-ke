# required libraries
from matplotlib import pyplot as plt
from nltk import PorterStemmer, SnowballStemmer, LancasterStemmer
from pathlib import Path
import networkx as nx
import nltk
import numpy as np
import os
import pandas as pd
import random


class GraphMaker:
    def __init__(self, stopwordsPath, stemmer) -> None:
        with open(Path(stopwordsPath), 'r') as handler:
            parsed_stopwords = [line.rstrip() for line in handler.readlines()]

        self.stopwords = set(parsed_stopwords)

        if stemmer == 'POR':
            self.stemmer = PorterStemmer()
        elif stemmer == 'SNO':
            self.stemmer = SnowballStemmer('english')
        elif stemmer == 'LAN':
            self.stemmer = LancasterStemmer()
        else:
            raise Exception('Unknown stemmer! Available options: POR (Porter), SNO (Snowball, english), LAN (Lancaster)')

    def buildGraph(self, text) -> nx.Graph: # (andre) ho tolto parametro filename, non usato
        k = 0.5

        font_size = 26
        words_in_text = []
        is_noun = lambda word, pos: pos[:2] == 'NN' and not (word in self.stopwords)
        is_adjective_or_verb = lambda word, pos: (pos[:2] == 'JJ' or pos[:2] == 'VB') and not (word in self.stopwords)

        for sent in text.split('.')[:-1]:
            tokenized = nltk.word_tokenize(sent)
            nouns_adj_ver = [word for (word, pos) in nltk.pos_tag(tokenized) if
                             (is_noun(word, pos) or is_adjective_or_verb(word, pos))]
            words_in_text.append(' '.join([word for word in nouns_adj_ver if not (
                    len(word) <= 2)]))  # nouns_in_text becomes a list of Strings, each of which acts as list of nouns

        words_list = []  # will determine the numbers and the id of the rows

        for sent in words_in_text:  # deleting duplicates
            temp = sent.split(' ')
            for word in temp:  # stemming and adding stems
                if self.stemmer.stem(word) not in words_list:
                    if not (self.stemmer.stem(word) == ""):
                        words_list.append(self.stemmer.stem(word))

        df = pd.DataFrame(np.zeros(shape=(len(words_list), 2)), columns=['Stems', 'neighbors'])
        df['Stems'] = words_list

        for sent in text.split('.'):
            tokens = nltk.word_tokenize(sent)
            stemmed_sent = [self.stemmer.stem(word) for (word, pos) in nltk.pos_tag(tokens) if
                            is_noun(word, pos) or is_adjective_or_verb(word, pos)]

            for stem in words_list:
                if stem in stemmed_sent:
                    ind = df[df['Stems'] == stem].index[0]
                    df['neighbors'][ind] = [ss for ss in stemmed_sent if not (ss == stem)]
                    try:
                        float(str(df['neighbors'][ind]))
                        if (df['neighbors'][ind] == 0):
                            print("This stem will cause the mistake: " + stem)
                    except Exception:
                        ops = 0
                        del ops
        G = nx.Graph()
        color_map = []
        for i in range(len(df)):
            G.add_node(df['Stems'][i])
            color_map.append('blue')
            # print("questa iterazione considera come vicini: "+ df['neighbors'][i])
            try:
                for word in df['neighbors'][i]:
                    G.add_edges_from([(df['Stems'][i], word)])
            except TypeError:
                ops = 0
                del ops
                # print("\n"+"I found problems in row containing stem "+ str(df['Stems'][i]))
        return G


def printGraph(G) -> None:
    k = 0.5
    font_size = 26
    pos = nx.spring_layout(G, k)

    d = nx.degree(G)
    node_sizes = []
    for i in d:
        _, value = i
        node_sizes.append(value)

    color_list = []
    for i in G.nodes:
        # value = nltk.pos_tag([i])[0][1]
        value = d[i]

        # red nodes = the most important
        if (value <= 10):
            color_list.append('yellow')
        elif (value <= 20):
            color_list.append('red')
        else:
            color_list.append('blue')

    plt.figure()#figsize=(40, 40))
    nx.draw(G, pos, node_size=[(v + 1) * 200 for v in node_sizes], with_labels=True, node_color=color_list,
            font_size=font_size)
    plt.show()


def localPageRankApprox(G) -> dict:
    nodes_list = G.nodes
    n = len(nodes_list)
    alpha = 0.85
    computed_centralities = {}  # dizionario per le centralità calcolate
    r = 10
    for u in nodes_list:
        # initializing for t=0
        PR_u = np.zeros(r + 1)
        PR_u[0] = 1
        layers = np.empty(r + 1, dtype=dict)
        layers[0] = {}
        layers[0][u] = 1
        inf_dict = np.empty(r + 1, dtype=dict)  # at each lvl there will be inf_t(x,u) with key x and value u
        inf_dict[0] = {}
        inf_dict[0][u] = 1
        for t in range(1, r + 1):
            layers[t] = {}
            inf_dict[t] = {}
            # initializing layer t: we put in layer t only neighbors of nodes in layers[t-1] che non sono già stati visitati: elif caso 0
            last_visited = layers[t - 1]
            for v in last_visited:
                for w in G.neighbors(v):
                    # se w è un vicino che non è ancora stato visitato
                    if t > 1:
                        if w not in last_visited and not w in layers[t - 2]:
                            layers[t][w] = 1
                    else:
                        layers[t][w] = 1
            # ho inizializzato il layer t
            for v in layers[t]:
                s = 0
                for w in layers[t - 1]:
                    if w in G.neighbors(v):
                        s += inf_dict[t - 1][w]
                inf_dict[t][v] = (1 / G.degree[v]) * s

            totSum = sum(layers[t][key] for key in layers[t])
            PR_u[t] = PR_u[t - 1] + ((1 - alpha) * (alpha ** t) / n) * totSum
        computed_centralities[u] = PR_u[r]
    return computed_centralities


def improvedEstimateLCC(G, p) -> dict:
    edges_list = G.edges
    S = []  # sample of edges
    # sono grafi piccoli e stanno in memoria
    t_S = {}
    deg = {}
    cc = {}
    # random.seed(os.urandom) # TODO fix ? a me fa TypeError sta riga :( (andre)
    random.seed(65165)  # intanto ho messo un seed a caso stile gatto sulla tastiera
    for v in G.nodes:
        t_S[v] = 0
        deg[v] = 0

    for edge in edges_list:
        u = edge[0]
        v = edge[1]
        deg[v] += 1
        deg[u] += 1

        N_S = []  # set of vertices
        N_s_v = []
        for x in G.neighbors(v):
            if (x, v) in S:
                N_s_v.append(x)
            elif (v, x) in S:
                N_s_v.append(x)

        for x in G.neighbors(u):
            if (u, x) in S:
                if x in N_s_v:
                    N_S.append(x)
            elif (x, u) in S:
                if x in N_s_v:
                    N_S.append(x)

        for c in N_S:
            t_S[c] += 1
            t_S[u] += 1
            t_S[v] += 1

        if random.random() < p:
            S.append(edge)

    for v in G.nodes:

        cc[v] = (1 / p ** 2) * 2 * t_S[v] / (deg[v] * (deg[v] - 1))  # TODO qui può capitare division by zero (andre)
        # l'unico caso possibile è che ci sia un nodo con deg[v] == 1
        # cercando su internet ho trovato che di solito viene definito come undefined oppure 0
        # non mi pare Vandin abbia specificato qualcosa a riguardo nelle slide (?)
        """
                # takes into account nodes with deg[v] = 1, avoid division by 0
                cc[v] = (1 / p ** 2) * 2 * t_S[v] / (deg[v] * (deg[v] - 1)) if deg[v] > 1 else 0 
        """

    return cc


def approximateClosenessCentrality(G, k) -> dict:
    s = {}
    # random.seed(os.urandom)  # TODO come riga 165
    random.seed(854181)  # anche qui ho usato un comodo gatto da tastiera
    nodes_list = list(G.nodes)
    n = len(nodes_list)
    c = {}
    for v in nodes_list:
        s[v] = 0
    for i in range(1, k + 1):
        v_i = random.choice(nodes_list)
        # solve single source shortest path for v_i
        for v in nodes_list:
            s[v] += (len(nx.shortest_path(G, v_i, v)) - 1)

    for v in nodes_list:
        c[v] = k * (n - 1) / (n * s[v])

    return c
