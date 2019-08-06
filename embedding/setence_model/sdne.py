# _*_ coding:utf-8 _*_
from walk_core_model import core_model
from util_tool import read_graph,get_node_information
import numpy as np
import tensorflow as tf
from keras.layers import Input,Dense
from keras import Model
from keras.regularizers import l1_l2
from keras import backend as K

class sdne(core_model):

    def __init__(self, Graph, per_vertex, walk_length, window_size, dimension_size, work,alpha,beta,epochs,batch_size,verbose):
        super().__init__(Graph, per_vertex, walk_length, window_size, dimension_size, work)
        self.alpha=alpha
        self.beta=beta
        self.batch_size=batch_size
        self.epochs=epochs
        self.verbose=verbose
        self.pred_all_nodes=self.all_nodes
        self.idx2node, self.node2idx = get_node_information(self.pred_all_nodes)
        self.W,self.W_ = self.generator_adjacency_matrix(self.pred_all_nodes)
        self.L=self.generator_L(self.W_)

    def generator_adjacency_matrix(self,all_nodes):
        numNodes=len(all_nodes)
        W=np.zeros((numNodes,numNodes))
        W_=np.zeros((numNodes,numNodes))

        for start_vertex in all_nodes:
            start_rank=self.node2idx[start_vertex]
            for end_vertex in list(self.G.neighbors(start_vertex)):
                end_rank=self.node2idx[end_vertex]
                weight=self.G[start_vertex][end_vertex].get('weight',1.0)
                W[start_rank][end_rank]=weight
                W_[start_rank][end_rank]=weight
                W_[end_rank][start_rank]=weight

        return W,W_

    def generator_L(self,W):
        D = np.zeros_like(W)

        for i in range(len(W)):
            D[i][i] = np.sum(W[i])
        L = D - W

        return L

    def first_nd(self,alpha):
        def first_loss(y_true, y_pred):
            loss = 2 * alpha * tf.linalg.trace(tf.matmul(tf.matmul(y_pred, y_true, transpose_a=True), y_pred))
            return loss / tf.to_float(K.shape(y_pred)[0])

        return first_loss

    def second_nd(self,beta):
        def second_loss(y_true,y_pred):
            b_=np.ones_like(y_true)
            b_[y_true!=0]=beta
            loss=K.sum(K.square((y_true-y_pred)*b_),axis=-1)
            return K.mean(loss)

        return second_loss

    def encoder(self,x,hidden_size_list,l1,l2):
        for i in range(len(hidden_size_list)-1):
            x=Dense(units=hidden_size_list[i],activation='relu',kernel_regularizer=l1_l2(l1,l2))(x)
        y=Dense(units=hidden_size_list[-1],activation='relu',kernel_regularizer=l1_l2(l1,l2),name='encode')(x)

        return y

    def decoder(self,y,hidden_size_list,l1,l2):
        for i in reversed(range(len(hidden_size_list)-1)):
            y=Dense(units=hidden_size_list[i],activation='relu',kernel_regularizer=l1_l2(l1,l2))(y)
        x=Dense(units=self.batch_size,activation='relu',kernel_regularizer=l1_l2(l1,l2),name='decode')(y)

        return x

    def creat_model(self,hidden_size_list,l1,l2):
        adjacency_matrix=Input(shape=(self.numNodes,))
        L=Input(shape=(None,))
        x=adjacency_matrix

        y=self.encoder(x,hidden_size_list,l1,l2)
        x_=self.decoder(y,hidden_size_list,l1,l2)

        model=Model(inputs=[adjacency_matrix,L],outputs=[x_,y])
        emb=Model(inputs=adjacency_matrix,outputs=y)

        self.model=model
        self.embedding_model=emb

        return model,emb

    def generator_data(self):
        all_nodes=self.pred_all_nodes
        start_rank=0
        end_rank=min(self.batch_size,self.numNodes)

        while True:
            batch_nodes=all_nodes[start_rank:end_rank]
            node_index_list=[self.node2idx[node] for node in batch_nodes]

            batch_W=self.W[node_index_list,:]
            print(batch_W.shape)
            batch_L=self.L[node_index_list][:,node_index_list]
            print(batch_L.shape)

            input_=[batch_W,batch_L]

            yield (input_,input_)

            start_rank = end_rank
            end_rank += self.batch_size
            end_rank = min(end_rank, self.numNodes)

            if end_rank==self.numNodes:
                start_rank=0
                end_rank=min(self.batch_size,self.numNodes)
                np.random.shuffle(all_nodes)

    def train(self,hidden_size_list,l1,l2,log_dir):
        model,emb=self.creat_model(hidden_size_list=hidden_size_list,l1=l1,l2=l2)
        model.compile('adam',[self.second_nd(self.beta),self.first_nd(self.alpha)])
        self.model.fit_generator(
            self.generator_data(),
            steps_per_epoch=self.numNodes//self.batch_size,
            epochs=self.epochs,
            callbacks=self.model_prepare(log_dir),
            verbose=self.verbose
        )

    def get_embeddings(self):
        embeddings={}
        pred_embeddings=self.embedding_model(inputs=self.W)

        for i,embedding in pred_embeddings:
            embeddings[self.idx2node[i]]=embedding

        return embeddings

if __name__=='__main__':
    Graph=read_graph()
    sden_model=sdne(
        Graph=Graph,
        dimension_size=128,
        per_vertex=100,
        walk_length=10,
        window_size=5,
        work=1,
        beta=5,
        alpha=1e-6,
        verbose=1,
        epochs=1000,
        batch_size=512
    )

    sden_model.train(
        log_dir='logs/0/',
        hidden_size_list=[256,128],
        l1=1e-5,
        l2=1e-4
    )
    # embeddings=sden_model.get_embeddings()
    #
    # from evaluate import evaluate_tools
    # eval_tool=evaluate_tools(embeddings)
