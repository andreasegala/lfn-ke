from pathlib import Path
from tqdm import tqdm
import commons.parse
import commons.graph
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns


def precision_recall(kw_true, kw_pred):
    # we can use sets since words (i.e., graph nodes) are all uniques
    tp = set(kw_true) & set(kw_pred)
    fp = set(kw_pred) - set(kw_true)
    fn = set(kw_true) - set(kw_pred)

    # P = TP / (TP + FP)
    precision = len(tp) / (len(tp) + len(fp)) if len(tp) + len(fp) > 0 else 0
    # R = TP / (TP + FN)
    recall = len(tp) / (len(tp) + len(fn)) if len(tp) + len(fn) > 0 else 0

    return precision, recall


def centrality_print_scores(parsed_articles, centrality_name, approximation, run_name) -> None:
    attribute_name = ''
    centrality_results = {}

    # step 1: compute the desired centrality
    print('Computing centrality...')
    # PR = PageRank
    if centrality_name == 'PR':
        if approximation == 0:
            attribute_name = 'PR'
            for art in tqdm(parsed_articles):
                centrality_results = nx.pagerank(art.graph)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
        else:
            attribute_name = 'approxPR'
            for art in tqdm(parsed_articles):
                r = 5
                centrality_results = commons.graph.localPageRankApprox(art.graph, r)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
    # LCC = Local Clustering Coefficient
    elif centrality_name == 'LCC':
        if approximation == 0:
            attribute_name = 'LCC'
            for art in tqdm(parsed_articles):
                centrality_results = nx.clustering(art.graph, art.graph.nodes)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
        else:
            attribute_name = 'approxLCC'
            for art in tqdm(parsed_articles):
                centrality_results = commons.graph.improvedEstimateLCC(art.graph, 0.5)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
    # CC = Closeness Centrality
    elif centrality_name == 'CC':
        if approximation == 0:
            attribute_name = 'CC'
            for art in tqdm(parsed_articles):
                centrality_results = nx.closeness_centrality(art.graph)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
        else:
            attribute_name = 'approxCC'
            for art in tqdm(parsed_articles):
                centrality_results = commons.graph.approximateClosenessCentrality(art.graph, 10)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
    # BC = Betweenness Centrality
    elif centrality_name == 'BC':
        if approximation == 0:
            attribute_name = 'BC'
            for art in tqdm(parsed_articles):
                centrality_results = nx.betweenness_centrality(art.graph)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
        else:
            attribute_name = 'approxBC'
            for art in tqdm(parsed_articles):
                centrality_results = nx.betweenness_centrality(art.graph, k=5)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
    else:
        raise Exception('Unknown centrality! Available options:\n\t- BC (Betweenness)\n\t- PR (PageRank)\n\t- CC (Closeness)\n\t- LCC (Local Clustering Coefficient)')

    # step 2: create folder /experiments/run_name
    print('Saving the results...')
    run_directory = './experiments/' + run_name
    Path(run_directory).mkdir(parents=True, exist_ok=True)

    centrality_file = run_directory + '/' + attribute_name + '.csv'

    # step 3: compute the scores and save them in a dataframe
    # dataframe setup
    column_names = ['ID', 'P@5', 'P@10', 'P@15', 'P@20', 'P@tot', 'R@5', 'R@10', 'R@15', 'R@20', 'R@tot']
    scores_df = pd.DataFrame(np.zeros(shape=(len(parsed_articles), len(column_names))), columns=column_names)

    # fill in the df
    for i in tqdm(range(len(parsed_articles))):  # will also be the row number
        article = parsed_articles[i]
        scores_df['ID'][i] = article.docID
        sorted_nodes = sorted(nx.get_node_attributes(article.graph, attribute_name).items(), key=lambda x: x[1])
        sorted_words = [node[0] for node in sorted_nodes]
        for j in [5, 10, 15, 20]:
            p_j, r_j = precision_recall(article.kw, sorted_words[:j])
            scores_df['P@' + str(j)][i] = p_j
            scores_df['R@' + str(j)][i] = r_j

        p_tot, r_tot = precision_recall(article.kw, sorted_words)
        scores_df['P@tot'][i] = p_tot
        scores_df['R@tot'][i] = r_tot

    # step 4: convert df into csv file
    scores_df.to_csv(centrality_file, index=False)


def significant_differences(centrality_list, approximation, metric_name, run_name) -> None:
    # step 1: paths and folders + df initialization
    run_dir_path = './experiments/' + run_name + '/'

    column_names = []
    for cc in centrality_list:
        if approximation == 0:
            # generate name of the column, which will also be the name of the file
            column_names.append(cc)
        elif approximation == 1:
            column_names.append('approx' + cc)
        else:  # both columns
            column_names.append(cc)
            column_names.append('approx' + cc)

    df = pd.DataFrame(columns=column_names)

    # step 2: for each centrality in centrality_list fill in the df with the given metric
    for c_name in column_names:
        # find the correct file
        c_path = Path(run_dir_path + c_name).with_suffix('.csv')

        # from file extract the column named metric as a list
        temp_df = pd.read_csv(c_path)
        column_to_keep = temp_df[metric_name]
        # add the column to needed df
        df[c_name] = column_to_keep
        del temp_df

    # step 3: perform the statistic analysis

    # step 3.1: create and print a boxplot for mean and variance
    sns.boxplot(x='variable', y='value', data=pd.melt(df)).set(xlabel='Centralities', ylabel=metric_name)

    # step 3.2: Tukey HSD test to assess significance of differences: each group is the column of a df
    # prepare list of groups
    group_list = []

    for c_name in column_names:
        group_list.append(df[c_name])

    res = scipy.stats.tukey_hsd(*group_list)
    print(res)


def average_metric(metric_name, run_name) -> pd.DataFrame:
    # step 0: find the path to the right directory and list of the files to scan + df initialization
    run_dir_path = './experiments/' + run_name + '/'
    file_names = ['PR', 'approxPR', 'CC', 'approxCC', 'BC', 'approxBC', 'LCC', 'approxLCC']

    df = pd.DataFrame(columns=file_names)

    # step 1: from each .csv file extract the column of the given metric and compute its average: save the value in the df
    for nn in file_names:
        # find the correct file
        c_path = Path(run_dir_path + nn).with_suffix('.csv')

        temp_df = pd.read_csv(c_path)
        column_to_keep = temp_df[metric_name]
        # add the average to needed df
        df[nn] = [np.mean(column_to_keep)]
        del temp_df

    return df
