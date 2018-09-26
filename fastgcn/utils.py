import networkx as nx
import numpy as np
import time
import datetime

def run_time_record(logging):
    def _run_time_record(function):
        def inner(*args, **kwargs):
            start = time.time()
            ans = function(*args, **kwargs)
            print("%s %s(%s seconds)" % (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), logging, round(time.time() - start, 3)))
            return ans
        return inner
    return _run_time_record

def construct_feed_dict(inputs,labels,supports,placeholders,len):
    feed_dict = dict()
    feed_dict.update({placeholders['inputs']: inputs})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['supports'][i]: supports[i] for i in range(len)})
    return feed_dict

class DataProcessor(object):
    def __init__(self,**kwargs):
        self._load_data(**kwargs)
        self.DiGraph = self._build_graph()

    def __call__(self, *args, **kwargs):
        return self.dataset_nodes,self.dataset_edges

    @run_time_record("Loaded data successfully!")
    def _load_data(self,**kwargs):
        self.dataset_nodes = []
        self.dataset_edges = []
        self.classes_dictionary = {}
        allowed_kwargs = {"data_content_file_path", "data_cites_file_path", "dataset_name","class_num"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, "Invaild keyword argument: " + kwarg
        data_content_file_path = kwargs.get("data_content_file_path")
        self.data_content_file_path = data_content_file_path
        data_cites_file_path = kwargs.get("data_cites_file_path")
        self.data_cites_file_path = data_cites_file_path
        dataset_name = kwargs.get("dataset_name")
        self.dataset_name = dataset_name
        if kwargs.get("class_num") == None:
            self.class_num = 7
        if dataset_name == "cora":
            with open(self.data_content_file_path, "r") as data_file:
                index = 0
                for line in data_file:
                    temp_line = line.strip().split("\t")
                    self.dataset_nodes.append({"paper_id": temp_line[0], "word_attributes": np.array(list(map(int, temp_line[1:-1]))), "class_label": temp_line[-1]})
                    if index < self.class_num and self.classes_dictionary.get(str(temp_line[-1]),"N") == "N":
                        temp = np.full(self.class_num,0)
                        temp[index] = 1
                        self.classes_dictionary[temp_line[-1]] = temp
                        index += 1
            with open(self.data_cites_file_path, "r") as data_file:
                for line in data_file:
                    temp_line = line.strip().split("\t")
                    self.dataset_edges.append({"cited_id": temp_line[0], "citing_id": temp_line[1]})
        elif dataset_name is None:
            print("Invaild dataset name")
        else:
            pass

    @run_time_record("Created directed graph successfully!")
    def _build_graph(self,dataset_nodes=None,dataset_edges=None):
        if dataset_nodes == None:
            dataset_nodes = self.dataset_nodes
        if dataset_edges == None:
            dataset_edges = self.dataset_edges
        DiGraph = nx.DiGraph(name=self.dataset_name)
        if self.dataset_name == "cora":
            for node in dataset_nodes:
                DiGraph.add_node(node.get("paper_id"),**node)
            for i,edge in enumerate(dataset_edges):
                DiGraph.add_edge(u_of_edge=edge.get("citing_id"),v_of_edge=edge.get("cited_id"))
        elif self.dataset_name == None:
            print("Invaild dataset name")
        else:
            pass
        return DiGraph

    @run_time_record("writed graphML successfully!")
    def write_graphML(self,write_graphML_path):
        dataset_nodes = self._general_dataset_nodes(keyword="word_attributes",separator="")
        DiGraph = self._build_graph(dataset_nodes=dataset_nodes)
        self._write_graphML(graph=DiGraph,write_graphML_path=write_graphML_path)

    @run_time_record("writed gexf successfully!")
    def write_gexf(self,write_gexf_path):
        dataset_nodes = self._general_dataset_nodes(keyword="word_attributes",separator="")
        DiGraph = self._build_graph(dataset_nodes=dataset_nodes)
        self._write_gexf(graph=DiGraph,write_gexf_path=write_gexf_path)

    def get_data(self):
        x = []
        y = []
        for node in self.dataset_nodes:
            x.append(node["word_attributes"])
            y.append(self.classes_dictionary.get(node["class_label"]))
        return np.array(x),np.array(y)

    def get_adjacent_matrix(self):
        return nx.adjacency_matrix(self.DiGraph,weight=1).toarray()

    def _general_dataset_nodes(self,keyword,separator):
        dataset_nodes = self.dataset_nodes.copy()
        for node in dataset_nodes:
            self._dictionary_list_to_str(dict=node, keyword=keyword, separator=separator)
        return dataset_nodes

    def _write_gexf(self,graph,write_gexf_path,encoding="utf-8"):
        nx.write_gexf(G=graph,path=write_gexf_path,encoding=encoding)

    def _write_graphML(self,graph,write_graphML_path,encoding="utf-8"):
        nx.write_graphml(G=graph,path=write_graphML_path,encoding=encoding)

    def _dictionary_list_to_str(self,dict,keyword,separator):
        return dict.update({keyword:separator.join(str(i) for i in dict.get(keyword))})


