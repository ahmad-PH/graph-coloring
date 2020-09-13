class GraphDataset(Dataset):
    def __init__(self, foldername):
        file_names = os.listdir(foldername)
        counter = 0
        while "{}.graph".format(counter) in file_names:
            counter += 1
        self.len = counter
        self.foldername = foldername

    def __getitem__(self, i):
        return Graph.load('{}/{}.graph'.format(self.foldername, i))

    def __len__(self):
        return self.len

    def __iter__(self):
        return iter(self.__getitem__(i) for i in range(self.len))

class GraphDatasetEager(GraphDataset):
    def __init__(self, foldername):
        super().__init__(foldername)
        self.graphs = [Graph.load('{}/{}.graph'.format(foldername, i)) for i in range(self.len)]

    def __getitem__(self, i):
        return self.graphs[i]
