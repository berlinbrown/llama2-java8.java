# Pytorch examples

import torch
import numpy as np
import time

randint = torch.randint(-100, 100, (6, ))

randint

print(randint)

tensor = torch.tensor([
    [0.2, 0.2],
    [1.2, 1.1],
    [1.4, 1.5]
])

print(tensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
