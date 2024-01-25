# required libraries
from matplotlib import pyplot as plt
from nltk import PorterStemmer, SnowballStemmer, LancasterStemmer
from pathlib import Path
import networkx as nx
import nltk
import numpy as np
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
            raise Exception(
                'Unknown stemmer! Available options:\n\t- POR (Porter)\n\t- SNO (Snowball, english)\n\t- LAN (Lancaster)')

    def buildGraph(self, text) -> nx.Graph:
        sentences = []
        # helpers functions
        is_noun = lambda word, pos: pos.startswith('NN') and word not in self.stopwords
        is_adjective_or_verb = lambda word, pos: pos.startswith(('JJ', 'VB')) and word not in self.stopwords

        # split the text in sentences and tokenize each
        for sentence in text.split('.'):
            tokenized_sentence = nltk.word_tokenize(sentence)
            nouns_adj_verbs = [word for word, pos in nltk.pos_tag(tokenized_sentence) if
                               is_noun(word, pos) or is_adjective_or_verb(word, pos)]
            sentences.append(' '.join([word for word in nouns_adj_verbs if len(word) > 2]))

        # get the word list
        word_list = {self.stemmer.stem(word) for sentence in sentences for word in sentence.split() if len(self.stemmer.stem(word))}

        # create a pd.DataFrame for each stem to store its neighbors
        stem_neighbors_df = pd.DataFrame({
            'stem': list(word_list),
            'neighbors': ''
        })

        # retrieve neighbors for each stem
        for sentence in text.split('.'):
            tokens = nltk.word_tokenize(sentence)
            stemmed_sentence = [self.stemmer.stem(word) for word, pos in nltk.pos_tag(tokens) if
                                is_noun(word, pos) or is_adjective_or_verb(word, pos)]

            for stem in word_list:
                if stem in stemmed_sentence:
                    neighbors = {s for s in stemmed_sentence if s != stem}
                    row_idx = stem_neighbors_df[stem_neighbors_df['stem'] == stem].index[0]
                    stem_neighbors_df.at[row_idx, 'neighbors'] = neighbors

        # build the graph
        word_graph = nx.Graph()

        for idx, row in stem_neighbors_df.iterrows():
            stem = row['stem']
            neighbors = row['neighbors']
            word_graph.add_node(stem)
            word_graph.add_edges_from([(stem, neighbor) for neighbor in neighbors])

        return word_graph


def printGraph(G) -> None:
    k = 0.4
    font_size = 28
    pos = nx.spring_layout(G, k)

    d = nx.degree(G)
    node_sizes = []
    for i in d:
        _, value = i
        node_sizes.append(value)

    color_list = []
    for i in G.nodes:
        value = d[i]
        if value <= 10:
            color_list.append('yellow')
        elif value <= 20:
            color_list.append('red')
        else:
            color_list.append('blue')

    plt.figure(figsize=(40, 40))
    nx.draw(G, pos, node_size=[(v + 1) * 200 for v in node_sizes], with_labels=True, node_color=color_list, edge_color='gray', font_size=font_size)
    plt.show()


def localPageRankApprox(G,r) -> dict:
    nodes_list = G.nodes
    n = len(nodes_list)
    alpha = 0.85
    computed_centralities = {}  # dict for the computed centrality
    
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
            # initializing layer t
            # we put in layer t only neighbors of nodes in layers[t-1] which have not already been visited
            last_visited = layers[t - 1]
            for v in last_visited:
                for w in G.neighbors(v):
                    # if w is a neighbor that has not yet been visited
                    if t > 1:
                        if w not in last_visited and not w in layers[t - 2]:
                            layers[t][w] = 1
                    else:
                        layers[t][w] = 1
            # init layer t
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
    # they're graph little enough to fit in memory
    t_S = {}
    deg = {}
    cc = {}
    random.seed(65165)
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
        # takes into account nodes with deg[v] = 1, avoid division by 0
        # 0 because these nodes are surely not involved in any triangle
        cc[v] = (1 / p ** 2) * 2 * t_S[v] / (deg[v] * (deg[v] - 1)) if deg[v] > 1 else 0

    return cc


def approximateClosenessCentrality(G, k) -> dict:
    s = {}
    random.seed(854181)
    nodes_list = list(G.nodes)
    n = len(nodes_list)
    c = {}
    for v in nodes_list:
        s[v] = 0
    for i in range(1, k + 1):
        v_i = random.choice(nodes_list)
        # solve single source shortest path for v_i
        for v in nodes_list:
            path_length = len(nx.shortest_path(G, v_i, v))
            s[v] += (path_length - 1)

    for v in nodes_list:
        c[v] = k * (n - 1) / (n * s[v])

    return c
