import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

# 1.导入DDP需要用到的库
# 用于支持多进程操作
import torch.multiprocessing as mp
# 用于在分布式训练中对数据集进行采样，确保每个进程看到不同的数据子集
from torch.utils.data.distributed import DistributedSampler
# 用于将模型分布到多个 GPU 上并行训练
from torch.nn.parallel import DistributedDataParallel as DDP
# 分别用于初始化分布式训练组和清理分布式训练组
from torch.distributed import init_process_group, destroy_process_group
import os


# 2.增加DDP相关参数设置的函数
def ddp_setup(rank, world_size):
    """
    设置分布式数据并行(DDP)环境，在训练前需要调用完成设置

    Args:
        rank (int): 当前进程的唯一标识符(通常为 0 到 world_size-1)。
        world_size (int): 参与分布式训练的总进程数（通常等于 GPU 的数量）。
    """
    # 设置主节点的地址，这里使用本地主机（localhost）作为主节点
    os.environ["MASTER_ADDR"] = "localhost"

    # 设置主节点的端口号，确保所有进程使用相同的端口进行通信
    # 主节点（MASTER_ADDR 和 MASTER_PORT）负责协调所有进程之间的通信，
    # 分布式训练需要所有进程能够互相通信，因此这些设置必须一致
    os.environ["MASTER_PORT"] = "12355"

    # 设置当前进程使用的 GPU 设备，rank 通常对应 GPU 的索引
    torch.cuda.set_device(rank)

    # 初始化进程组，用于分布式训练
    # - backend="nccl": 使用 NCCL 后端，NCCL 是 NVIDIA 提供的用于多 GPU 通信的高性能库
    # - rank=rank: 当前进程的标识符
    # - world_size=world_size: 总进程数
    init_process_group(backend="nccl", rank=rank, world_size=world_size)



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        # 3.将模型封装到DDP中，用于后面对应GPU进程中模型参数的分发
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        # 4.由于model现在是DDP的封装，不能直接通过model.state_dict()来直接获取模型的参数了
        ckp = self.model.module.state_dict()  
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)

            # 5.由于每个进程中的模型参数是一样的，所以只需要保存一个进程上的模型参数
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """
    Transform the dataset into dataloader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # 6.由于已经指定了采样器，所以将shuffle设置为false
        sampler=DistributedSampler(dataset)  # 7.保证数据集会被切分到不同的进程中，并且不会产生重复样本
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)  # 8.需要初始化进程组
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()  # 9.确保在训练完成后正确关闭分布式进程组，释放资源并停止后台进程
 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    # 10.训练部分需要使用mp.spawn完成各基层参数的传递
    world_size = torch.cuda.device_count()  # 获取GPU数量
    mp.spawn(main,  # 每个进程执行的函数
             args=(world_size, args.save_every, args.total_epochs, args.batch_size),  # 传递给 main 的参数,其中参数rank会被自动分配
             nprocs=world_size  # 进程数
            )
