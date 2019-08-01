# _*_ coding:utf-8 _*_
from deepwalk import DeepWalk
from line import Line
from node2vec import node2vec
from util_tool import read_graph
from evaluate import evaluate_tools

def deep_walk_run():
    Graph = read_graph('wiki/Wiki_edgelist.txt')

    deepwalk = DeepWalk(
        Graph=Graph,
        per_vertex=80,
        walk_length=10,
        window_size=5,
        dimension_size=128,
        work=4
    )

    embeddings = deepwalk.transform()
    eval = evaluate_tools(embeddings=embeddings,label_path='wiki/Wiki_labels.txt')
    eval.plot_embeddings()

def line_run():
    from util import read_graph
    import os
    print(os.getcwd())
    Graph = read_graph('wiki/Wiki_edgelist.txt')
    line = Line(
        Graph=Graph,
        dimension_size=128,
        per_vertex=100,
        walk_length=10,
        window_size=5,
        work=1,
        negative_ratio=1,
        batch_size=128,
        log_dir='logs/0/',
        epoch=100,
    )
    embeddings = line.transform()
    from evaluate import evaluate_tools
    tool = evaluate_tools(embeddings,label_path='wiki/Wiki_labels.txt')
    tool.plot_embeddings()

def node2vec_run():
    Graph = read_graph('wiki/Wiki_edgelist.txt')

    node_vec = node2vec(
        Graph=Graph,
        per_vertex=80,
        walk_length=10,
        window_size=5,
        dimension_size=128,
        work=1,
        p=0.25,
        q=4
    )

    embeddings = node_vec.transform()
    eval_tool = evaluate_tools(embeddings,label_path='wiki/Wiki_labels.txt')
    eval_tool.plot_embeddings()

def test(build_name):
    if build_name=='deepwalk':
        deep_walk_run()
    elif build_name=='line':
        line_run()
    elif build_name=='node2vec':
        node2vec_run()
    elif build_name=='all':
        deep_walk_run()
        line_run()
        node2vec_run()

if __name__=='__main__':
    test('all')