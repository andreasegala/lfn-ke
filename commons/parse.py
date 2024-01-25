# required libraries
from pathlib import Path
from tqdm import tqdm
import commons.graph
import json
import networkx as nx
import nltk
import random
import re


def update_allowed_forbidden_files(graphmaker, min_nodes):
    # get all the filenames
    filenames = Path("../data/elsevier_oaccby_corpus/json/").glob("*.json")

    # lists to store allowed and forbidden files
    allowed_files = []

    # required keys for a document to be valid
    required_keys = ['docId', 'abstract']
    required_metadata_keys = ['keywords', 'subjareas']

    # for each file, check if there are all the required keys and mark the file as allowed
    # and check that the largest connected component has at least min_nodes nodes
    for filename in tqdm(filenames, total=40001):
        # load JSON file
        file = open(filename, 'r')
        json_file = json.load(file)
        file.close()

        # check the keys and the largest connected component of the graph
        # and marks the file as allowed or forbidden
        # it saves whole relative paths
        is_allowed = True
        if all(key in json_file for key in required_keys) and all(key in json_file['metadata'] for key in required_metadata_keys):
            article_graph = graphmaker.buildGraph(json_file['abstract'])
            largest_connected_component = len(max(list(nx.connected_components(article_graph)), key=len))
            if largest_connected_component < min_nodes:
                is_allowed = False
        else:
            is_allowed = False

        # if the file is allowed, append it to the corresponding list
        if is_allowed:
            allowed_files.append(str(filename))

    # write allowed files into a .txt file
    allowed_writer = open(Path('resources/allowed_files.txt'), 'w')
    allowed_writer.write('\n'.join(allowed_files))
    allowed_writer.close()
    print(f'There are {len(allowed_files)} allowed files.')


class Article:
    def __init__(self, docID, abstract, categories, keywords, filename, graphmaker) -> None:
        self.docID = docID
        cleanAbstract = re.sub("['\",]", '', abstract)
        self.abstract = cleanAbstract
        self.categories = categories
        self.filename = filename

        # take only largest connected component
        article_graph = graphmaker.buildGraph(abstract)
        largest_connected_component = max(list(nx.connected_components(article_graph)), key=len)
        self.graph = article_graph.subgraph(largest_connected_component).copy()

        # store keywords already stemmed
        kws = set()
        for keyword in keywords:
            tokens = nltk.word_tokenize(keyword)
            for token in tokens:
                kws.add(graphmaker.stemmer.stem(token))
        self.kw = list(kws)

    def printGraph(self):
        commons.graph.printGraph(self.graph)


def parse_and_sample(sample_size, graphmaker) -> list:  # list of Article objs
    sampled_articles = []
    random.seed(3101960)

    # sample sample_size paths from the allowed ones
    with open(Path('resources/allowed_files.txt'), 'r') as handler:
        allowed_paths = [Path(line.rstrip()) for line in handler.readlines()]
    sampled_paths = random.choices(allowed_paths, k=sample_size)

    # create Article objs
    for path in tqdm(sampled_paths):
        with open(path, 'r') as handler:
            # load JSON file
            json_file = json.load(handler)

            # extract data of interest
            docID = json_file['docId']
            abstract = json_file['abstract']
            keywords = json_file['metadata']['keywords']
            categories = json_file['metadata']['subjareas']
            filename = path.name

            # create Article and append to sampled list
            new_article = Article(docID, abstract, categories, keywords, filename, graphmaker)
            sampled_articles.append(new_article)

    return sampled_articles

