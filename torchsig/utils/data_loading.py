from torch.utils.data import DataLoader
from torchsig.utils.random import Seedable

class WorkerSeedingDataLoader(DataLoader, Seedable):
    """
    A Custom DaaLoader for torchsig that seeds workers differently on worker init based on a shared initial seed;
    """

    def __init__(self, dataset, **kwargs):
        DataLoader.__init__(self, dataset, **kwargs)
        Seedable.__init__(self, **kwargs)
        if self.worker_init_fn:
            raise ValueError("No worker_init_fn should be given to WorkerSeedingDataLoader; it will set it's own worker_init_fn.")
        self.worker_init_fn = self.init_worker_seed

    def init_worker_seed(self, worker_id):
        from torch.utils.data import get_worker_info
        get_worker_info().dataset.seed(int(self.random_generator.random()*100 + 1) * (worker_id + 1))