import torch
import os
from torch.utils.data import Dataset, DataLoader

from graph import Graph
from exact_coloring import find_chromatic_number
from manual_emb_main import learn_embeddings
from utility import save_to_file

class GraphDataset(Dataset):
    def __init__(self, foldername):
        file_names = os.listdir(foldername)
        counter = 0
        while "{}.graph".format(counter) in file_names:
            counter += 1
        self.len = counter
        self.foldername = foldername

    def __getitem__(self, i) -> Graph:
        return Graph.load('{}/{}.graph'.format(self.foldername, i))

    def __len__(self):
        return self.len

    def __iter__(self):
        return iter(self.__getitem__(i) for i in range(self.len))

class GraphDatasetEager(GraphDataset):
    def __init__(self, foldername):
        super().__init__(foldername)
        self.graphs = [Graph.load('{}/{}.graph'.format(foldername, i)) for i in range(self.len)]

    def __getitem__(self, i) -> Graph:
        return self.graphs[i]


def generate_embeddings_for_dataset(ds_folder: str, verbose=False):
    os.mkdir(os.path.join(ds_folder, 'embeddings'))
    ds = GraphDataset(ds_folder)
    for i, graph in enumerate(ds):
        embeddings, results = learn_embeddings(graph, 10)
        if verbose: print('i: {}, results: {}'.format(i, results))
        torch.save(embeddings, '{}/embeddings/{}.pt'.format(ds_folder, i))


def generate_optimal_solutions_for_dataset(ds_folder: str, verbose=False):
    os.mkdir(os.path.join(ds_folder, 'solutions'))
    ds = GraphDataset(ds_folder)
    for i, graph in enumerate(ds):
        chromatic_number, solution = find_chromatic_number(graph)
        if verbose: print('i: {}, chromatic_number: {},\nsol: {}\n'.format(i, chromatic_number, solution))
        save_to_file(solution, '{}/solutions/{}.txt'.format(ds_folder, i))
