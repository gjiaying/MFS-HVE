import sys
import os
sys.path.append('..')
import math
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd
from scipy.linalg import circulant
from torchtext.vocab import GloVe
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

class Mtb(fewshot_re_kit.framework.FewShotREModel):
    """
    Use the same few-shot model as the paper "Matching the Blanks: Distributional Similarity for Relation Learning".
    """
    
    def __init__(self, sentence_encoder, image_encoder, use_dropout=True, combiner="max"):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder, image_encoder)
        self.multi_hidden = 768+128
        self.text_hidden = 256
        self.image_hidden = 256
        self.object_hidden = 256
        
        self.fcs = nn.Linear(512, 256)
        self.fcq = nn.Linear(512, 256)
        self.fc = nn.Linear(self.multi_hidden, self.multi_hidden)
        self.drop = nn.Dropout()
        self.softmax = nn.Softmax()
        self.use_dropout = use_dropout
        self.layer_norm = torch.nn.LayerNorm(sentence_encoder.bert.config.hidden_size * (2 if sentence_encoder.cat_entity_rep else 1))
        self.combiner = combiner
        self.textfc = nn.Linear(768, self.text_hidden, bias=True)
        self.imagefc = nn.Linear(512, self.image_hidden, bias=True)
        self.imgfc = nn.Linear(256, 128, bias=True)
        self.txtfc = nn.Linear(256, 512, bias=True)
        self.objfc = nn.Linear(256, 256, bias=True)
        
        self.fc1 = nn.Linear(2560, 1280, bias=True)
        self.fc2 = nn.Linear(1280, 512, bias=True)
        self.fc3 = nn.Linear(2560, 1280)
        
        self.fc4 = nn.Linear(1280, 1280, bias=True)
        self.fc5 = nn.Linear(6400 ,1280, bias=True)
        
        
        # for feature-level attention
        self.shots = 1
        self.conv1 = nn.Conv2d(1, 32, (self.shots, 1), padding=(self.shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (self.shots, 1), padding=(self.shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (self.shots, 1), stride=(self.shots, 1))
        
        
        #for object-level attention
        self.embedding_glove = GloVe(name='6B', dim=50)
        self.objectfc = nn.Linear(200, 256, bias=True) # pre-trained embedding to 256
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(self.object_hidden)
        #self attention
        
        
        self.fcq_query = nn.Linear(self.text_hidden, self.text_hidden)
        self.fcv_query = nn.Linear(self.text_hidden, self.text_hidden)
        self.fck_query = nn.Linear(self.text_hidden, self.text_hidden)
        self.fcq_support = nn.Linear(self.text_hidden, self.text_hidden)
        self.fcv_support = nn.Linear(self.text_hidden, self.text_hidden)
        self.fck_support = nn.Linear(self.text_hidden, self.text_hidden)
        
        self.fcqo_query = nn.Linear(256, 256)
        self.fcvo_query = nn.Linear(self.object_hidden, self.object_hidden)
        self.fcko_query = nn.Linear(self.object_hidden, self.object_hidden)
        self.fcqo_support = nn.Linear(256, 256)
        self.fcvo_support = nn.Linear(self.object_hidden, self.object_hidden)
        self.fcko_support = nn.Linear(self.object_hidden, self.object_hidden)
        #img-level attention
        self.fcqi_query = nn.Linear(self.image_hidden, self.image_hidden)
        self.fcvi_query = nn.Linear(self.image_hidden, self.image_hidden)
        self.fcki_query = nn.Linear(self.image_hidden, self.image_hidden)
        self.fcqi_support = nn.Linear(self.image_hidden, self.image_hidden)
        self.fcvi_support = nn.Linear(self.image_hidden, self.image_hidden)
        self.fcki_support = nn.Linear(self.image_hidden, self.image_hidden)

    def __dist__(self, x, y, dim):
        sqdist = torch.sum((x - y) ** 2, dim=-1)
        squnorm = torch.sum(x ** y, dim=-1)
        sqvnorm = torch.sum(x ** y, dim=-1)
        u = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
        v = torch.sqrt(u ** 2 - 1)
        return torch.log(u + v).sum(dim)
        #return (x * y).sum(dim)
        
        
        

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    
    def index_to_image(self, image, img_list):
        data_array = []
        for b in range(image.size()[0]):
            data_array.append(img_list[image[b]])  
        return data_array
    
    def word_embedding(self, words):
        if len(words) == 0:
            words.append('None')
            words.append('None')
            words.append('None')
            words.append('None')
        elif len(words) == 1:
            words.append('None')
            words.append('None')
            words.append('None')
        elif len(words) == 2:
            words.append('None')
            words.append('None')
        elif len(words) == 3:
            words.append('None')
        else:
            words = words[0:4]
        
        #for i in range(2):
          #  embedding_matrix[i] = self.embedding_glove[words[i]]
        
        embedding_matrix = np.concatenate([self.embedding_glove[words[0]],self.embedding_glove[words[1]], self.embedding_glove[words[2]],self.embedding_glove[words[3]]])
        #embedding_matrix = self.embedding_glove[words[0]]
        return embedding_matrix
        
    
    def object_detection(self, objects, img_id):
        embedding_matrix = np.zeros((len(img_id), 200))
        for i in range(len(img_id)):
            obj = objects[img_id[i]]
            embedding_matrix[i] = self.word_embedding(obj)
        
            
    
        return torch.from_numpy(embedding_matrix).cuda()
    
    
    
    def circulants(self, tensor, dim):
        S = tensor.shape[dim]
        tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
        return tmp.unfold(dim, S, 1).flip((-1,))

    
    
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            #mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)
        scores = F.softmax(scores, dim=-1)
    
        if dropout is not None:
            scores = dropout(scores)
        
        output = torch.matmul(scores, v)
        return output
        

    def forward(self, support, query, s_img, q_img, N, K, total_Q, img_list, objects):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        img_list: whole list of images
        '''
        
        support = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query) # (B * total_Q, D)
        print(support.size())
        support_img_id = self.index_to_image(s_img, img_list)
        query_img_id = self.index_to_image(q_img, img_list)
        support_img = self.image_encoder(support_img_id)
        query_img = self.image_encoder(query_img_id)
        img_hidden_size = support_img.size(-1)
        
        object_support = self.object_detection(objects, support_img_id) #B*N*K, 100
        object_support = self.objectfc(object_support.float())
        object_query = self.object_detection(objects, query_img_id) #B*N*K, 100
        object_query = self.objectfc(object_query.float())
       
        
        
        
        if self.use_dropout:
            support = self.drop(support)
            query = self.drop(query)
            support_img = self.drop(support_img)
            query_img = self.drop(query_img)
        #print(support)
        support = self.layer_norm(support)
        query = self.layer_norm(query)
        
        
        support_text = self.textfc(support.float())
        query_text = self.textfc(query.float())
        support_img = self.imagefc(support_img.float())
        query_img = self.imagefc(query_img.float())
        
        
        #object-level attention
        support_text_q = self.fcq_support(support_text)
        support_text_v = self.fcv_support(support_text)
        support_text_k = self.fck_support(support_text)
        query_text_q = self.fcq_query(query_text)
        query_text_v = self.fcv_query(query_text)
        query_text_k = self.fck_query(query_text)
        
        
        support_object_q = self.fcqo_support(object_support)
        support_object_v = self.fcqo_support(object_support)
        support_object_k = self.fcqo_support(object_support)
        query_object_q = self.fcqo_query(object_query)
        query_object_v = self.fcvo_query(object_query)
        query_object_k = self.fcko_query(object_query)
        #img-level attention
        support_img_q = self.fcqi_support(support_img)
        support_img_v = self.fcvi_support(support_img)
        support_img_k = self.fcki_support(support_img)
        query_img_q = self.fcqi_query(query_img)
        query_img_v = self.fcvi_query(query_img)
        query_img_k = self.fcki_query(query_img)
        
        
        #support_text_o = self.layernorm(support_text+self.dropout(self.attention(support_object_q, support_text_k, support_text_v, self.object_hidden)))
        #query_text_o = self.layernorm(query_text+self.dropout(self.attention(query_object_q, query_text_k, query_text_v, self.object_hidden)))
        support_text_o = self.softmax(self.fcq(torch.cat((support_text, support_object_q),1))) * object_support
        query_text_o = self.softmax(self.fcq(torch.cat((query_text, query_object_q),1))) * object_query
        
        support_img_att = self.layernorm(support_img+self.dropout(self.attention(support_text_q, support_img_k, support_img_v, self.image_hidden)))
        query_img_att = self.layernorm(query_img+self.dropout(self.attention(query_text_q, query_img_k, query_img_v, self.image_hidden)))
        #support_img_att = self.softmax(self.fcq(torch.cat((support_img, support_text_q),1))) * support_img
        #query_img_att = self.softmax(self.fcq(torch.cat((query_img, query_text_q),1))) * query_img
        
        
        
        #hidden_size = support_text.size(-1)
        #img_size = support_img.size(-1)
        #object_size = support_text_o.size(-1)
        
        '''
        #circulant
        
        support = support_text
        query = query_text
        support_text_matrix = self.circulants(support, 1)
        support_img_matrix = self.circulants(support_img, 1)
        query_text_matrix = self.circulants(query, 1)
        query_img_matrix = self.circulants(query_img, 1)
        support_text_matrix_fuse = torch.matmul(support_text_matrix, support_img.transpose(0,1))
        support_img_matrix_fuse = torch.matmul(support_img_matrix, support.transpose(0,1))
        query_text_matrix_fuse = torch.matmul(query_text_matrix, query_img.transpose(0,1))
        query_img_matrix_fuse = torch.matmul(query_img_matrix, query.transpose(0,1))
    
        support_finals = torch.add(support_text_matrix_fuse, support_img_matrix_fuse)
        BNK = support_finals.size(0)
        query_finals = torch.add(query_text_matrix_fuse, query_img_matrix_fuse)
        BQK = query_finals.size(0)
        
        support_finals = support_finals.view(BNK, -1)
        query_finals = query_finals.view(BQK, -1)
        support_finals = self.fc1(support_finals)
        support_finals = self.fc2(support_finals)
        query_finals = self.fc3(query_finals)
        query_finals = self.fc2(query_finals)
        '''
        
        #original mtb and attention-based mtb
        support_text = self.txtfc(support_text)
        query_text = self.txtfc(query_text)
        support_img_att = self.imgfc(support_img_att)
        query_img_att = self.imgfc(query_img_att)
        support_text_o = self.objfc(support_text_o)
        query_text_o = self.objfc(query_text_o)
        
        
        support = torch.cat((support_text, support_img_att, support_text_o),1)
        query = torch.cat((query_text, query_img_att, query_text_o),1)
        support = self.fc(support)
        query = self.fc(query)
        
        support_final = support.view(-1, N, K, self.multi_hidden).unsqueeze(1)
        query_final = query.view(-1, total_Q, self.multi_hidden).unsqueeze(2).unsqueeze(2)
        
        
        
        #feature-level Attention
        support = torch.cat((support_text, support_img_att, support_text_o),1)
        query = torch.cat((query_text, query_img_att, query_text_o),1)
        
        
        support = support.view(-1, N, K, self.multi_hidden) #(B, N, K, D)hidden_size+img_hidden_size
        query = query.view(-1, N * 1, self.multi_hidden) #(B, N * Q, D)
        B = support.size(0) # Batch Size
        NQ = query.size(1) # Num of instances for each batch in the query set
        
        fea_att_score = support.view(B * N, 1, K, self.multi_hidden) # (B * N, 1, K, D)
        fea_att_score = F.relu(self.conv1(fea_att_score)) # (B * N, 32, K, D) 
        fea_att_score = F.relu(self.conv2(fea_att_score)) # (B * N, 64, K, D)
        fea_att_score = self.drop(fea_att_score)
        fea_att_score = self.conv_final(fea_att_score) # (B * N, 1, 1, D)
        fea_att_score = F.relu(fea_att_score)
        fea_att_score = fea_att_score.view(B, N, self.multi_hidden).unsqueeze(1) # (B, 1, N, D)
        

        
      #  support_final = support_final.view(-1, N, K, 768).unsqueeze(1) # (B, 1, N, K, D)
     #   query_final = query_final.view(-1, total_Q, 768).unsqueeze(2).unsqueeze(2) # (B, total_Q is NQ, 1, 1, D)
        
        #support_finals = torch.cat((support_final,support_final_o),4)
        #query_finals = torch.cat((query_final,query_final_o),4)
        
        logits = (support_final * query_final * fea_att_score).sum(-1) # (B, total_Q, N, K)
        #B 1 N K D * B NQ 1 1 D * B 1 1 N D
        
       
        
        # aggregate result
        if self.combiner == "max":
            combined_logits, _ = logits.max(-1) # (B, total, N)
        elif self.combiner == "avg":
            combined_logits = logits.mean(-1) # (B, total, N)
        else:
            raise NotImplementedError
        _, pred = torch.max(combined_logits.view(-1, N), -1)

        return combined_logits, pred
    
    
