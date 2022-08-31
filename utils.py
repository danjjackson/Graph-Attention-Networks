import enum

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

# Support for 3 different GAT implementations - we'll profile each one of these in playground.py
class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "saved_models/tutorial7"
BATCH_SIZE = 1

dataset = Planetoid(DATASET_PATH, 'Cora', split='public')
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)