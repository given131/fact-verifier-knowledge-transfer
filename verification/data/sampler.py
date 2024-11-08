from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler


class DistributedEvalSampler(DistributedSampler):
    """DistributedSampler for evaluation: no samples are added or dropped
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_size = len(self.dataset)
        self.num_samples = (
                self.total_size // self.num_replicas
                + int(dist.get_rank() < (self.total_size % self.num_replicas))
        )
