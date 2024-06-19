import ast
from astmonkey import visitors, transformers
import dgl
import networkx as nx
import numpy as np 
import os
import pickle
import torch

#not necessary (label Dict (which is a list) can be loaded dynamicall (in time))
# labelDict = ['Module()', 'Import()', "alias(name='time', asname=None)", 'Assign()', "Name(id='x', ctx=ast.Store())", 'Constant(value=1)', "Name(id='y', ctx=ast.Store())", 'Constant(value=8)', 'Constant(value=6)', 'While()', 'Compare()', "Name(id='y', ctx=ast.Load())", "Name(id='x', ctx=ast.Load())", 'If()', 'Constant(value=0)', 'Expr()', 'Call()', "Attribute(attr='sleep', ctx=ast.Load())", "Name(id='time', ctx=ast.Load())", 'Constant(value=2)', 'Constant(value=10)', 'AugAssign()', 'Add()', 'Constant(value=17)', 'AugAssign(op=ast.Sub())', 'AugAssign(op=ast.Add())', 'For()', "Name(id='i', ctx=ast.Store())", "Name(id='range', ctx=ast.Load())", 'Constant(value=7)', 'Constant(value=16)', 'Constant(value=12)', 'Constant(value=15)', 'Constant(value=19)', 'Constant(value=18)', 'Constant(value=4)', 'Constant(value=9)', 'Constant(value=5)', 'Constant(value=14)', 'Constant(value=13)', 'Constant(value=3)', 'Constant(value=11)']

class DataBatcher:
    def __init__(self, recalc_labels = False):
        self.data_folder = "data/"
        self.train_filenames_txt_file = self.data_folder + "train_files_paths.txt"
        with open(self.train_filenames_txt_file, 'r') as file:
            self.train_filenames = file.read().splitlines()
            #self.train_filenames = self.train_filenames[:50]
        self.test_filenames_txt_file = self.data_folder + "test_files_paths.txt"
        
        with open(self.test_filenames_txt_file, 'r') as file:
           self.test_filenames = file.read().splitlines()
           #self.test_filenames = self.test_filenames[:50]
        self.current_idx_train = 0
        self.current_idx_test = 0
        self.labels = {}
        if not recalc_labels:
            try:
                file_name = 'label_dict'
                with open (file_name, 'rb') as f:
                    self.labels = pickle.load(f)
                    print(f"Loading in labels from {file_name} pickle file.")
            except FileNotFoundError:
                self.__get_all_labels()
        else:
            self.__get_all_labels()
    
    def __get_all_labels(self):
        print ("Extracting all labels")
        all_filenames = self.train_filenames + self.test_filenames
        n = len(all_filenames)
        for count, filename in enumerate(all_filenames):
            print(f"Extracting labels from file {count}/{n}")
            with open(filename, 'r') as f :
                program = f.read()

            node = ast.parse(program)
            node = transformers.ParentChildNodeTransformer().visit(node)  

            visitor = visitors.GraphNodeVisitor()
            visitor.visit(node)
        
            N = nx.nx_pydot.from_pydot(visitor.graph)

            for i in N._node :         
                label = self.__get_label(i, N)
                if label not in self.labels: 
                    self.labels[label] = len(self.labels)

        print(f"There are {len(self.labels)} labels in total.")
        with open('label_dict', 'wb') as f:
            pickle.dump(self.labels, f)
        
    def get_batch(self, batch_size, train):    
        graphs = []
        labels = []
        filenames = []

        for i in range(batch_size): 
            if train:
                file = self.train_filenames[self.current_idx_train]
                self.current_idx_train = (self.current_idx_train + 1) % len(self.train_filenames)
            else:
                file = self.test_filenames[self.current_idx_test]
                self.current_idx_test = (self.current_idx_test + 1) % len(self.test_filenames)

            if "correct" in file:
                label = np.array([1])
            else:
                label = np.array([0])

            g, features = self.__codeToDgl(file)
            filenames.append(file)

            features = np.array(features)
            x = torch.FloatTensor(features)
            g.ndata['x'] = x

            graphs.append(g)        
            labels.append(label)
            
        return graphs, labels, filenames
    
    def __get_label(self, i, N):
        label = N._node[i]['label'][4:]
        label = label.replace(', type_comment=None', '')
        label = label.replace(', annotation=None', '')
        label = label.replace(', kind=None', '')
        label = label.replace(', returns=None', '')
        label = label.replace('type_comment=None', '')
        return label

    def __codeToDgl(self, filename): 

        with open(filename, 'r') as f :
            program = f.read()

        node = ast.parse(program)
        node = transformers.ParentChildNodeTransformer().visit(node)  

        visitor = visitors.GraphNodeVisitor()
        visitor.visit(node)
        
        N = nx.nx_pydot.from_pydot(visitor.graph)

        # mapping = {}
        features = []

        for i in N._node :         
            label = self.__get_label(i, N)
            num = self.labels[label]
            # mapping[i] = num -> Is this used??
            features.append(np.eye(len(self.labels))[num])

        #added !!
        #remapping before conversion
        mappingRelabel = {}
        for count, i in enumerate(N._node) :
            mappingRelabel[i] = count               
        N = nx.relabel_nodes(N, mapping=mappingRelabel)


        g = dgl.from_networkx(N)

        return g, features #also return label here
    
    def get_len_feature_space(self):
        return len(self.labels)



