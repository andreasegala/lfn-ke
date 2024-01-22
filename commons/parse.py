# required libraries
from pathlib import Path
import random
from tqdm import tqdm
import json
import networkx as nx
import re
import commons.graph
import nltk


def update_allowed_forbidden_files():
    # get all the filenames
    filenames = Path("../data/elsevier_oaccby_corpus/json/").glob("*.json")

    # lists to store allowed and forbidden files
    allowed_files = []
    forbidden_files = []

    # required keys for a document to be valid
    required_keys = ['docId', 'abstract']
    required_metadata_keys = ['keywords', 'subjareas']

    # TODO se volete tqdm si può togliere, l'avevo usato per vedere l'avanzamento (fa la progress bar)
    # for each file, check if there are all the required keys and mark the file as allowed
    # if not, mark the file as forbidden
    for filename in tqdm(filenames, total=40001):
        # load JSON file
        file = open(filename, 'r')
        json_file = json.load(file)
        file.close()

        # check the keys and mark as allowed or forbidden
        # it saves whole relative paths (see line 13)
        if all(key in json_file for key in required_keys) and all(key in json_file['metadata'] for key in required_metadata_keys):
            allowed_files.append(str(filename))
        else:
            forbidden_files.append(str(filename))

    # write allowed file into a .txt file
    allowed_writer = open(Path('resources/allowed_files.txt'), 'w')
    allowed_writer.write('\n'.join(allowed_files))
    allowed_writer.close()

    # write forbidden file into a .txt file
    # TODO (vedi linea 75) ma questo serve se salviamo gli allowed?
    forbidden_writer = open(Path('resources/forbidden_files.txt'), 'w')
    forbidden_writer.write('\n'.join(forbidden_files))
    forbidden_writer.close()


class Article:
    def __init__(self, docID, abstract, categories, keywords, filename, graphmaker) -> None:
        self.docID = docID
        cleanAbstract = re.sub("['\",]", '', abstract)
        self.abstract = cleanAbstract
        self.categories = categories
        self.filename = filename

        article_graph = graphmaker.buildGraph(abstract)
        largest_connected_component = max(list(nx.connected_components(article_graph)), key=len)
        self.graph = article_graph.subgraph(largest_connected_component).copy()

        kws = set()
        for keyword in keywords:
            tokens = nltk.word_tokenize(keyword)
            for token in tokens:
                kws.add(graphmaker.stemmer.stem(token))
        self.kw = list(kws)

    def setGraph(self, graphmaker) -> None:
        self.graph = graphmaker.buildGraph(self.abstract)

    def printGraph(self):
        commons.graph.printGraph(self.graph)

    def toString(self) -> str:
        return ''


def parse_and_sample(sample_size, graphmaker) -> list: # list of Article objs
    sampled_articles = []
    random.seed(3101960)

    # TODO (vedi linea 45) serve avere forbidden list se abbiamo allowed list?
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
            # TODO remove try catch
            try:
                new_article = Article(docID, abstract, categories, keywords, filename, graphmaker)
                sampled_articles.append(new_article)
            except:
                print("file", str(path))

    return sampled_articles

