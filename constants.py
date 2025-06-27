import matplotlib.pyplot as plt
import torch

SEED = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_PLT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
