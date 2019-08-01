# _*_ coding:utf-8 _*_
import tensorflow as tf
from keras.layers import Embedding,Input,Lambda
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau,TensorBoard,EarlyStopping,ModelCheckpoint
from keras import Model
from numpy import random
from backone_language_model import language_model
from backone_optimize import optimize_funcation

class core_model(object):

    def __init__(self,Graph,per_vertex,walk_length,window_size,dimension_size,work):
        self.G=Graph
        self.walk_epoch=per_vertex
        self.sentence_len=walk_length
        self.all_nodes=list(Graph.nodes())
        self.all_edges=list(Graph.edges())
        self.numEdges=Graph.number_of_edges()
        self.numNodes=Graph.number_of_nodes()
        self.dimension_size=dimension_size
        self.backone_model = language_model(
            dimension_size=dimension_size,
            window_size=window_size,
            work=work
        )
        self.optimize_fun=optimize_funcation()

    def unnormalized_transition_probability(self,t,v,p,q):
        '''
        :param v:目前所在顶点
        :param t: 上一次的节点
        :      x:下一步的目标节点
        :    x=t:d(tx)=0,1/p
        :  t-x=1:d(tx)=1,1
        :   else:d(tx)=2,1/q
        :return :edge_alias_table
        sampling weights:p(d(tx))*edge_weight
        '''

        unnormalized_probs=[]

        for x in self.G.neighbors(v):
            weight=self.G[v][x].get('weight',1.0)
            if x==t:
                unnormalized_probs.append(weight/p)
            elif self.G.has_edge(x,t):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight/q)
        norm_sum=sum(unnormalized_probs)
        all_probs=[float(un_prob)/norm_sum for un_prob in unnormalized_probs]

        edge_sample_table=self.optimize_fun.generate_alias_table(all_probs)

        return edge_sample_table

    def Preprocess_Modified_Weights(self,p,q):
        alias_nodes={}

        count=0
        for node in self.all_nodes:
            unnormalized_probs=[]
            for neighbor in self.G.neighbors(node):
                weight=self.G[node][neighbor].get('weight',1.0)
                unnormalized_probs.append(weight)

            norm_sum=sum(unnormalized_probs)
            all_probs=[float(un_probs)/norm_sum for un_probs in unnormalized_probs]
            alias_nodes[node]=self.optimize_fun.generate_alias_table(all_probability=all_probs)

            count+=1

        alias_edges={}

        for edge in self.all_edges:
            alias_edges[edge]=self.unnormalized_transition_probability(edge[0],edge[1],p,q)

        self.alias_nodes=alias_nodes
        self.alias_edges=alias_edges

        import numpy as np
        np.save('walk_edges',alias_edges)


    def random_walk(self,start_vertex,is_edge_sampling=False):
        node_sentence=[start_vertex]
        now_walk_len=1

        while now_walk_len<self.sentence_len:
            now_node=node_sentence[-1]
            neighborhood_list=list(self.G.neighbors(now_node))
            if len(neighborhood_list)>0:
                if not is_edge_sampling:
                    next_node=random.choice(neighborhood_list)
                    node_sentence.append(next_node)
                else:
                    if len(node_sentence)==1:
                        next_node_rank=self.optimize_fun.alias_sample(prab=self.alias_nodes[now_node][0],alias=self.alias_nodes[now_node][1])
                        next_node=neighborhood_list[next_node_rank]
                        node_sentence.append(next_node)
                    else:
                        pre_node=node_sentence[-2]
                        edge=(pre_node,now_node)
                        next_node_rank=self.optimize_fun.alias_sample(self.alias_edges[edge][0],alias=self.alias_edges[edge][1])
                        next_node=neighborhood_list[next_node_rank]
                        node_sentence.append(next_node)
                now_walk_len+=1
            else:
                break

        return node_sentence

    def line_loss(self,y_true,y_pred):
        #在二阶有负样本,因为引入了-1的权重,故loss共用
        return -K.mean(K.log(K.sigmoid(y_true*y_pred)))

    def creat_line_model(self,order='second',lr=0.001):
        v_i = Input(shape=(1,))
        v_j = Input(shape=(1,))

        first_emb = Embedding(self.numNodes, self.dimension_size, name='first_emb')
        second_emb = Embedding(self.numNodes, self.dimension_size, name='second_emb')
        context_emb = Embedding(self.numNodes, self.dimension_size, name='context_emb')

        v_i_emb = first_emb(v_i)
        v_j_emb = first_emb(v_j)

        v_i_emb_second = second_emb(v_i)
        v_j_context_emb = context_emb(v_j)

        first = Lambda(lambda x: tf.reduce_sum(
            x[0] * x[1], axis=-1, keep_dims=False), name='first_order')([v_i_emb, v_j_emb])
        second = Lambda(lambda x: tf.reduce_sum(
            x[0] * x[1], axis=-1, keep_dims=False), name='second_order')([v_i_emb_second, v_j_context_emb])

        if order == 'first':
            output_list = [first]
        elif order == 'second':
            output_list = [second]
        else:
            output_list = [first, second]

        model = Model(inputs=[v_i, v_j], outputs=output_list)

        adam=Adam(lr=lr)
        model.compile(optimizer=adam,loss=self.line_loss)

        self.embedding_dict = {'first': first_emb, 'second': second_emb}

        return model

    def model_prepare(self,log_dir):
        tensorboard=TensorBoard(log_dir=log_dir)

        checkpoint=ModelCheckpoint(
            log_dir+'best_weights.h5',
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            period=1
        )

        earlystop=EarlyStopping(
            monitor='loss',
            patience=50
        )

        reduce_lr=ReduceLROnPlateau(
            monitor='loss',
            patience=1,
            factor=0.1
        )

        callback_list=[tensorboard,checkpoint,earlystop,reduce_lr]
        return callback_list

    def embdding_train(self,sentence_list):

        print('begin train embedding')
        print('loading...')

        model=self.backone_model.word2vec_on_train(sentence_list)

        print('train ending')

        embeddings={}
        for node in self.all_nodes:
            embeddings[node]=model.wv[node]

        return embeddings
