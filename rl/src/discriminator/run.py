import pandas as pd
from tqdm.auto import tqdm
from discriminator.config import *
tqdm.pandas()
import torch
from ts_tree_sitter.ts_parser import GoParser
from discriminator.model import BatchProgramClassifier
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class ASTNode(object):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        self.children = self.add_children()

    def is_leaf(self):
        return self.node.child_count == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        token=''
        if self.is_leaf():
            token = self.node.text.decode('utf-8')
            token = token.lower()
        return token

    def add_children(self):
        if self.is_str:
            return []
        children = self.node.children
        if self.token in ['FuncDef', 'If', 'While', 'DoWhile','Switch']:
            return [ASTNode(children[0][1])]
        elif self.token == 'For':
            return [ASTNode(children[c][1]) for c in range(0, len(children)-1)]
        else:
            return [ASTNode(child) for child in children]

    def compound(self):
        return self.token=="{"

class Pipeline:

    def __init__(self, s):
        self.original_program=s
        self.sources = None
        self.input_data = None

    def get_parsed_source(self) -> pd.DataFrame:
        processed_code = pd.DataFrame({'id': [0], 'code': [self.original_program], 'label': [0]}) 
        parser = GoParser()
        processed_code['code'] = processed_code['code'].apply(parser.parse)
        self.sources = processed_code

        return self.sources

    def get_sequences(self, node, sequence):
        current = ASTNode(node)
        if current.token != '':
            sequence.append(current.token)
        for child in node.children:
            self.get_sequences(child, sequence)
        if current.is_compound:
            sequence.append('End')

   
    def get_blocks(self, node, block_seq):
        children = node.children
        name = node.text.decode('utf-8')

        keywords = ['func', 'for', 'if']

        if children == []:
            block_seq.append(ASTNode(node))

        elif any(keyword in name for keyword in keywords):
            block_seq.append(ASTNode(node))

            for i in range(len(children)):
                child = children[i]
                if not any(keyword in name for keyword in keywords):
                    block_seq.append(ASTNode(child))
                self.get_blocks(child, block_seq)
        elif '{' in name:
            block_seq.append(ASTNode(node))
            for child in node.children:
                if not any(keyword in name for keyword in keywords):
                    block_seq.append(ASTNode(child))
                self.get_blocks(child, block_seq)
            block_seq.append(ASTNode('End'))
        else:
            for child in node.children:
                self.get_blocks(child, block_seq)

    def generate_block_seqs(self):
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load('discriminator/embedding/node_w2v_128').wv

        max_token = word2vec.vectors.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [word2vec.key_to_index[token] if token in word2vec.key_to_index else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            self.get_blocks(r.root_node, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        
        self.input_data=self.sources.copy()
        self.input_data['code'] = self.input_data['code'].apply(trans2seq)

    def run(self):
        self.get_parsed_source()
        self.generate_block_seqs()
        
        tmp = self.input_data.iloc[0: 1]
        data = []
        for _, item in tmp.iterrows():
            data.append(item[1])
            
        return data


class Cluster:
    @classmethod
    def cluster(cls,program_string=None):

        if not isinstance(program_string, str):
            return "Error: program_string should be a string type."
        
        model_input=Pipeline(program_string).run()
        
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load("discriminator/embedding/node_w2v_128").wv
        embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
        embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

        MAX_TOKENS = word2vec.vectors.shape[0]
        EMBEDDING_DIM = word2vec.vectors.shape[1]

        model=BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
        if USE_GPU:
            model.cuda()
        
        model.load_state_dict(torch.load('discriminator/saved_model/best_model.bin'))
        model.eval()

        with torch.no_grad(): 
            output = model(model_input)
        
        max_value, max_index = torch.max(output[0], dim=0)

        return max_index.item()+1,output[0].tolist()


