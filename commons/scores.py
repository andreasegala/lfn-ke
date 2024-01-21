import networkx as nx
import commons.parse
import commons.graph
from pathlib import Path
import pandas as pd
import numpy as np 

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

    attribute_name = ""
    centrality_results={}
    # Step 1: Compute the desired centrality: switch case
    if centrality_name == "PR":

        if approximation ==0:

            attribute_name = "PR"
            for art in parsed_articles:
                centrality_results = nx.pagerank(art.graph)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
        else:
            attribute_name = "approxPR"
            for art in parsed_articles:
                centrality_results = commons.graph.localPageRankApprox(art.graph)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
    elif centrality_name == "LCC":

        if approximation ==0:

            attribute_name= "LCC"
            for art in parsed_articles:
                centrality_results = nx.improvedEstimateLCC(art.graph,1)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
        else:
            attribute_name= "approxLCC"
            for art in parsed_articles:
                centrality_results = commons.graph.improvedEstimateLCC(art.graph, 0.5)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)

    elif centrality_name=="CC":

        if approximation ==0:

            attribute_name= "CC"
            for art in parsed_articles:
                centrality_results = nx.closeness_centrality(art.graph)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
        else:
            attribute_name= "approxCC"
            for art in parsed_articles:
                centrality_results = commons.graph.approximateClosenessCentrality(art.graph, 10)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
    elif centrality_name== "BC":
        if approximation ==0:

            attribute_name= "BC"
            for art in parsed_articles:
                centrality_results = nx.betweenness_centrality(art.graph)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
        else:
            attribute_name= "approxBC"
            for art in parsed_articles:
                centrality_results = commons.graph.betweenness_centrality(art.graph, k= 20)
                nx.set_node_attributes(art.graph, centrality_results, attribute_name)
    else:
        raise Exception('Unknown centrality! Available options: BC (Betweenness), PR (PageRank), CC (Closenss), LCC (Local Clustering Coefficient)')

            

    #Step 2: Create folder /experiments/run_name
    run_directory = 'experiments/'+ run_name
    Path(run_directory).mkdir(exist_ok=True)

    centrality_file = run_directory+'/'+attribute_name+".csv"

    #Step 3: Compute the scores and save them in a dataframe

    #dataframe setup
    column_names = ['ID','P@5','P@10', 'P@15','P@20','P@tot','R@5','R@10', 'R@15','R@20','R@tot']
    scores_df = pd.DataFrame(np.zeros(shape=(len(parsed_articles),len(column_names))), columns = column_names)

    #fill in the df

    for i in range(0,len(parsed_articles)): #will also be the row number
        
        article = parsed_articles[i]
        scores_df['ID'][i] = article.docID
        sorted_nodes = sorted(nx.get_node_attributes(article.graph, attribute_name).items() ,key = lambda x: x[1])
        sorted_words = [node[0] for node in sorted_nodes]
        for j in [5, 10, 15, 20]:
            p_j, r_j = precision_recall(article.kw, sorted_words[:j])
            scores_df['P@'+str(j)][i]= p_j
            scores_df['R@'+str(j)][i]= r_j
        
        p_tot, r_tot = precision_recall(article.kw, sorted_words)
        scores_df['P@tot'][i]= p_tot
        scores_df['R@tot'][i]= r_tot

    #Step 4: convert df into csv file
        scores_df.to_csv(centrality_file, index=False)




            




