'''
Run the graph embedding methods on Karate graph and evaluate them on 
graph reconstruction and visualization. Please copy the 
gem/data/karate.edgelist to the working directory
'''
import matplotlib.pyplot as plt
from time import time

from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr

from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne     import SDNE
from argparse import ArgumentParser


if __name__ == '__main__':
    ''' Sample usage
    python run_karate.py -node2vec 1
    '''
    parser = ArgumentParser(description='Graph Embedding Experiments on Homo Graphs')

    # Specify whether the edges are directed
    isDirected = True

    models = []
    # Load the models you want to run
    models.append(GraphFactorization(d=64, max_iter=1000, eta=1 * 10**-4, regu=1.0, data_set='sbm'))
    models.append(HOPE(d=64, beta=0.01))
    models.append(LaplacianEigenmaps(d=64))
    models.append(LocallyLinearEmbedding(d=64))
    models.append(
        node2vec(d=64, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1, data_set='sbm')
    )
    models.append(SDNE(d=64, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[500, 300,], rho=0.3, n_iter=30, xeta=0.001,n_batch=500,
                    modelfile=['enc_model.json', 'dec_model.json'],
                    weightfile=['enc_weights.hdf5', 'dec_weights.hdf5']))

    import os
    dirpath = 'data/homo_graph'
    gfiles = [gfile for gfile in os.listdir(dirpath) if os.path.splitext(gfile)[1] == '.csv']

    results = {}
    from collections import namedtuple
    Stats = namedtuple("stats", "MAP prec_curv err err_baseline")

    for gfile in gfiles:
        G = graph_util.loadGraphFromEdgeListTxt(
            os.path.join(dirpath, gfile), directed=isDirected, has_prefix=True)
        G = G.to_directed()
        print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
        results[gfile] = {}
        for embedding in models:    
            t1 = time()
            # Learn embedding - accepts a networkx graph or file with edge list
            Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
            print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
            # Evaluate on graph reconstruction
            MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
            results[gfile][embedding._method_name] = Stats(MAP, prec_curv, err, err_baseline)
            #---------------------------------------------------------------------------------
            print(("\tMAP: {} \t preccision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
            #---------------------------------------------------------------------------------
            # Visualize
            