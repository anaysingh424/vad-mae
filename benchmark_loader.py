import time
import torch  # type: ignore
from data.train_dataset import AbnormalDatasetGradientsTrain  # type: ignore
from configs.configs import get_configs_avenue  # type: ignore

class Args:
    def __init__(self):
        conf = get_configs_avenue()
        for k, v in conf.items():
            setattr(self, k, v)
        self.num_workers = 4
        self.device = 'cpu'

if __name__ == '__main__':
    args = Args()
    dataset = AbnormalDatasetGradientsTrain(args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4)
    start = time.time()
    for i, data in enumerate(loader):
        if i >= 5: break
        print(f"Batch {i} loaded")
    print(f"Time for 5 batches with 4 workers: {time.time() - start:.2f}s")
