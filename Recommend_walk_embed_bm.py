import random
from hdt import HDTDocument
import numpy as np
from numpy import dot
from numpy.linalg import norm
from gensim.parsing.preprocessing import preprocess_string
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import networkx as nx
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

all_args = argparse.ArgumentParser()
all_args.add_argument("-th", "--threshold", required=True,
                      help="Threshold for similarity between entity(author) embedding")
all_args.add_argument("-bth", "--bm25_threshold", required=True,
                      help="Threshold for BM25 ranking")
all_args.add_argument("-hp", "--hop", required=True,
                      help="Hop number for graph walk")
all_args.add_argument("-sd", "--seed", required=True,
                      help="Path to [seed file].txt")
all_args.add_argument("-cd", "--candidate", required=True,
                      help="Path to [candidate file].txt")
all_args.add_argument("-gd", "--standard", required=True,
                      help="Path to [standard file].hdt")
all_args.add_argument("-d", "--dir", required=False, default=os.path.dirname(os.getcwd()),
                      help="Directory to read all needed files and to store all results. Default is directory of this python file")
args = vars(all_args.parse_args())
if not os.path.isdir(args['dir']):
    raise ValueError("Please input correct directory path. -d/--dir [path_to_dir]")
if not os.path.isfile(args['seed']):
    raise ValueError("Please input correct path for seed file")
if not os.path.isfile(args['candidate']):
    raise ValueError("Please input correct path for candidate file")
if not os.path.isfile(args['standard']):
    raise ValueError("Please input correct path for standard file")
if float(args['threshold']) < 0.0 or float(args['threshold']) > 1.0:
    raise ValueError("Please input valid threshold for similarity between entity(author) embedding")
if int(args['hop']) < 0 or int(args['hop']) > 3:
    raise ValueError("Please input hop number for graph walk in range [1-3]")
if int(args['bm25_threshold']) < 0:
    raise ValueError("Please input threshold for BM25 ranking >0")


def graphwalk(G: nx.Graph, hop: int, seed, cand_list): # graph walk implementation
    walk_authors = set()
    if G.has_node(seed.replace("https://makg.org/entity/", "")):
        temp_set = set(
            nx.single_source_shortest_path(G, seed.replace("https://makg.org/entity/", ""), cutoff=(2 * hop)).keys())
        temp_set = set([str("https://makg.org/entity/" + t) for t in temp_set])
        walk_authors = cand_list.intersection(temp_set)
    return walk_authors


def read_seed_dataset(path): # load all the seed datasets from file
    seed_datasets = set()
    with open(path) as seed_file:
        for line in seed_file:
            seed_datasets.add(line.strip())
    return seed_datasets


def read_candidate_dataset(path): # load all the candidate datasets from file 
    dataset_list = set()
    with open(path) as seed_file:
        for line in seed_file:
            dataset_list.add(line.strip())
    return dataset_list


def reduce_candidates(seeds, candidates, standard_hdt): # if use large seed/candidates input file and wanna use not all seed/candidate for experiemnt, this function will reduce the size of seed/candidate set 
    new_cands = set()
    for seed in seeds:
        (triples, cardinality) = standard_hdt.search_triples_bytes(seed, "", "")
        for subj, pred, obj in triples:
            s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
            if o in candidates:
                new_cands.add(o)
    return new_cands


def get_authors_based_on_datasets(dataset_list, dataset_author_hdt): # get the author list of one dataset list
    author_list = set()
    for dataset in dataset_list:
        (triples, cardinality) = dataset_author_hdt.search_triples_bytes("", "", dataset)
        for subj, pred, obj in triples:
            s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
            author_list.add(s)
    return author_list


def get_seed_author(seed_dataset, coauthor_hdt): # get the authors of seed datset (or one dataset)
    seed_authors = set()
    (triples, cardinality) = coauthor_hdt.search_triples_bytes("", "", seed_dataset)
    for subj, pred, obj in triples:
        s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
        seed_authors.add(s)
    return seed_authors


def cosine_np(a, b): # cosine similarity function
    return dot(a, b) / (norm(a) * norm(b))


def clean_candidate_with_ent_embed(seed_author, candidate_set, ent_embed, threshold=0.0): # remove the author which cannot meet author embedding similarity threshold from author set
    new_set = []
    seed_vec = ent_embed[seed_author]
    for candidate in candidate_set:
        candidate_vec = ent_embed[candidate]
        sim = cosine_np(seed_vec, candidate_vec)
        if sim >= threshold:
            new_set.append(candidate)
    return new_set


def get_standard(dataset, standard_hdt): # get standard linked dataset set for given seed dataset
    standards = set()
    (triples, cardinality) = standard_hdt.search_triples_bytes(dataset, "", "")
    for subj, pred, obj in triples:
        s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
        standards.add(o)
    return standards


def read_vec_dict(path): # read dictionary of author entity vector for quick index 
    vec_dict = {}
    with open(path) as rfile:
        for line in rfile:
            spline = line.split()
            if spline[1].startswith("entity/"):
                vec_dict[str(spline[1].replace("entity/", ""))] = int(spline[0])
    rfile.close()
    return vec_dict


def authors_to_embed_dict(author_list, vec_dict, fp): # transfer a set of author into index-set of author vectors
    embed_dict = {}
    for author in author_list:
        embed_dict[author] = fp[vec_dict[author.replace("https://makg.org/entity/", "")]]
    return embed_dict


def authors_to_datasets(author_list, dataset_author_hdt): # find datset set based on given author list/set
    dataset_list = set()
    for author in author_list:
        (triples, cardinality) = dataset_author_hdt.search_triples_bytes(author, "", "")
        for subj, pred, obj in triples:
            s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
            dataset_list.add(o)
    return dataset_list


def get_title(dataset, title_hdt): # get title of dataset from HDT file
    title = ""
    (triples, cardinality) = title_hdt.search_triples_bytes(dataset, "", "")
    for subj, pred, obj in triples:
        s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
        title = o.replace("\"", "")
        break
    return title


def get_abs(dataset, abs_hdt): # get abstract/description of dataset from HDT file
    title = ""
    (triples, cardinality) = abs_hdt.search_triples_bytes(dataset, "", "")
    for subj, pred, obj in triples:
        s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
        title = o.replace("\"", "")
        break
    return title


def get_datasets_title_desc_with_dict(dataset_list, title_hdt, abs_hdt): # prework for get title and abstract/description of dataset
    dataset_dict = {}
    for dataset in dataset_list:
        dataset_dict[dataset] = get_title(dataset, title_hdt) + " " + get_abs(dataset, abs_hdt)
    return dict(enumerate(list(dataset_dict.keys()))), list(dataset_dict.values())


def get_datasets_title_desc(dataset_list, title_hdt, abs_hdt): # combined function to get title and abstract/description of dataset
    dataset_dict = {}
    for dataset in dataset_list:
        dataset_dict[dataset] = get_title(dataset, title_hdt) + " " + get_abs(dataset, abs_hdt)
    return dataset_dict

def loop_build_graph(G: nx.Graph, step: int, node, coauthor_hdt): # prebuild the custom co-author network for all the authors from seed and candidate datasets. To reduce time consuming of graph walk
    if step % 2 == 0:
        (triples, cardinality) = coauthor_hdt.search_triples_bytes(
            node, "", "")
        for sub, _, obj in triples:
            s, o = sub.decode('utf-8'), obj.decode('utf-8')
            G.add_edge(s.replace("https://makg.org/entity/", ""), o.replace("https://makg.org/entity/", ""))
            step += 1
            if step < 3:
                loop_build_graph(G, step, o, coauthor_hdt)
    elif step % 2 == 0:
        (triples, cardinality) = coauthor_hdt.search_triples_bytes(
            "", "", node)
        for sub, _, obj in triples:
            s, o = sub.decode('utf-8'), obj.decode('utf-8')
            G.add_edge(s.replace("https://makg.org/entity/", ""), o.replace("https://makg.org/entity/", ""))
            step += 1
            if step < 3:
                loop_build_graph(G, step, s, coauthor_hdt)


def step(i, seed, standard_hdt, coauthor_hdt, Graph, candidate_author, all_author, threshold, result_w, result_w1,
         result_w2, seed_bm, bm25_index, cand_bm_dict, bm_t): # recommendatoin algorithem implementation
    walk_dataset = set() # empty output dataset set without embedding
    walk_dataset_embed = set() # empty output dataset set with embedding
    standards = get_standard(seed, standard_hdt) # get the standards linked datasets for given seed dataset
    seed_authors = get_seed_author(seed, coauthor_hdt) # get all the authors of seed dataset (author should be findable in MAKG co-author network)
    assert len(seed_authors) > 0 # make sure seed dataset contain at least one author
    for seed_author in seed_authors:
        walk_author = graphwalk(Graph, i, seed_author, candidate_author) # for each author of seed dataset, find hop-i authors
        walk_author = set([element for element in walk_author if element != None]) # make sure there is no None author
        walk_author = walk_author.intersection(candidate_author) # hop-i authors should be founded in the author set of all candidate datasets (reduce following useless computation)
        walk_dataset.update(authors_to_datasets(walk_author, coauthor_hdt)) # use hop-i authors to find datasets, and put these datasets into output dataset set without embedding
        walk_author = clean_candidate_with_ent_embed(seed_author, walk_author, all_author, threshold) # remove hop-i authors which cannot meet author entity embedding similarity threshold
        walk_dataset_embed.update(authors_to_datasets(walk_author, coauthor_hdt)) # use hop-i authors which meet embedding similarity threshold to find datasets and put into output dataset set with embedding
    # only graph walk method evaluation
    G = float(len(standards)) # size of gold standard linked datasets
    correct_dataset = standards.intersection(walk_dataset) # True positive set = standard set intersect with recommended dataset set
    T = float(len(correct_dataset)) # size of True positive set
    N = float(len(walk_dataset)) # size of graph walk set
    precision = 0.0
    if N != 0.0:
        precision = T / N #precision
    recall = 0.0
    if G != 0.0:
        recall = T / G #recall
    result_w.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w.flush()
    # graph walk + author embedding similarity
    correct_dataset = standards.intersection(walk_dataset_embed)
    T = float(len(correct_dataset))
    N = float(len(walk_dataset_embed))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w1.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w1.flush()
    # graph walk + author embeeding similarity + BM25
    search_tokens = preprocess_string(seed_bm[seed])
    scores = bm25_index.get_scores(search_tokens) #create bm25 program with descriptive metadata of seed dataset as query
    bm_results = {}
    for i in range(len(cand_bm_dict)):
        bm_results[cand_bm_dict[i]] = scores[i] # for each recommended dataset, use BM25 program to calculate score between seed dataset and recommended dataset based on descriptive metadata
    bm_results = dict(sorted(bm_results.items(), key=lambda x: x[1], reverse=True)) # sort recommended dataset set with BM25 score
    bm_results = list(bm_results.keys())
    bm_results = set(bm_results[:(bm_t * int(len(standards)))]) # use BM25 threshold to remove recommended dataset which cannot meet threshold
    bm_results = bm_results.intersection(walk_dataset_embed)
    N = float(len(bm_results))
    correct_dataset = standards.intersection(bm_results)
    T = float(len(correct_dataset))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w2.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w2.flush()


if __name__ == '__main__':
    dir = str(args['out'])
    seed_percent = 1.0
    bm_t = int(args['bm25_threshold'])
    seed_path = dir + str(args['seed'])
    cand_path = dir + str(args['candidate'])
    standard_path = dir + str(args['standard'])
    #Read all needed data:
    vec_path = dir + "/mag_authors_2020_ComplEx_entity.npy"
    vec_dict_path = dir + "/authors_entities.dict"
    title_path = dir + "/Paper.hdt"
    des_path = dir + "/PaperAbs.hdt"
    coauthor_hdt = HDTDocument(dir + "/PaperAuthorAffiliations.hdt")

    # Prework:
    standard_hdt = HDTDocument(standard_path)
    seeds = read_seed_dataset(seed_path)
    candidates = read_candidate_dataset(cand_path)
    if seed_percent < 1.0:
        seeds = list(seeds)
        random.shuffle(seeds)
        seeds = set(seeds[:int(seed_percent * float(len(seeds)))])
        candidates = reduce_candidates(seeds, candidates, standard_hdt)
    print("Experiment with " + str(len(seeds)) + " seed and " + str(
        len(candidates)) + " candidate.")

    title_hdt = HDTDocument(title_path)
    des_hdt = HDTDocument(des_path)
    fp = np.memmap(vec_path, mode='r', dtype='float32', shape=(333085368, 100))
    vec_dict = read_vec_dict(vec_dict_path)
    all_seed_author = get_authors_based_on_datasets(seeds, coauthor_hdt)
    candidate_author = get_authors_based_on_datasets(candidates, coauthor_hdt)
    all_author = all_seed_author.union(candidate_author)
    Graph = nx.Graph()
    for author in tqdm(all_author, desc="Building hop1-3 coauthor graph"):
        loop_build_graph(Graph, 0, author, coauthor_hdt)
    all_author = authors_to_embed_dict(all_author, vec_dict, fp)
    seed_bm = get_datasets_title_desc(seeds, title_hdt, des_hdt)
    cand_bm_dict, cand_bm_titles = get_datasets_title_desc_with_dict(candidates, title_hdt, des_hdt)
    bm25_index = BM25Okapi([preprocess_string(title) for title in tqdm(cand_bm_titles, desc="Building bm25 index")])

    i = int(args['hop'])
    hop = str(i)

    threshold = float(args['threshold'])

    result_w = open(dir + "/hop" + hop + "_01.tsv", 'w')
    result_w1 = open(dir + "/hop" + hop + "_02_embed_" + str(threshold) + ".tsv",
                     'a+')
    result_w2 = open(dir + "/hop" + hop + "_03_embed_" + str(
        threshold) + "_bm25.tsv",
                     'a+')
    for seed in tqdm(seeds, desc="exp progress bar"):
        step(i, seed, standard_hdt, coauthor_hdt, Graph, candidate_author, all_author, threshold, result_w,
             result_w1, result_w2, seed_bm, bm25_index, cand_bm_dict, bm_t)
