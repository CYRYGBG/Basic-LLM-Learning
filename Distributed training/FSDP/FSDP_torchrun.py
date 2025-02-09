import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# 1. 增加包引入
import os
import functools
import torch.distributed as dist  # 用于多线程数据的处理
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

# 2. 增加FSDP相关参数设置的函数
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# 模型部分不进行修改
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# 3. 修改训练函数(仅训练一轮)使其符合FSDP的使用
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    # 新增用来存储分布式训练损失统计（损失总和+样本数）
    ddp_loss = torch.zeros(2).to(rank) 
    # 设置sampler的epoch，保证分布式训练中shuffle的正确性 
    if sampler:  
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据转移到指定的设备上
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    # 将不同进程中中的损失值和样本数进行相加以计算后续的平均损失
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    # 仅在主进程进行该操作
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

# 4. 修改测试函数使其符合FSDP的使用
def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    # 与train函数中的部分类似，记录损失值、预测正确的样本数和总样本数
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # 所有损失值求和
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测标签
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()  # 预测正确的样本数
            ddp_loss[2] += len(data)  # 样本数

    # 与train中功能类似
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))

# 5. 将模型封装在FSDP中并进行分布式训练
def fsdp_main(args):
    # 获取线程信息
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    setup(rank, world_size)

    # -----------数据处理和下载部分没有变化-----------
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
    # -----------数据处理和下载部分没有变化-----------

    # 与分布式
    sampler1 = DistributedSampler(dataset1,  # 用于采样的数据集
                                  rank=rank,  # 当前进程
                                  num_replicas=world_size,  # 参与分布式训练的进程的数量 
                                  shuffle=True)
    sampler2 = DistributedSampler(dataset2, 
                                  rank=rank, 
                                  num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 
                    'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 
                   'sampler': sampler2}
    cuda_kwargs = { 'num_workers': 2,       # 数据加载子进程数
                    'pin_memory': True,     # 启用锁页内存（加速GPU传输）
                    'shuffle': False}       # 分布式采样器已处理shuffle，此处必须禁用
    # 合并到训练和测试参数中
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # FSDP自动包装策略
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,    # 基于参数数量的自动分片策略
        min_num_params=100              # 参数超过100的模块会被分片
    )
    # 绑定当前进程到对应GPU
    torch.cuda.set_device(local_rank)

    # 记录初始化开始时间
    init_start_event = torch.cuda.Event(enable_timing=True)
    # 记录初始化结束时间
    init_end_event = torch.cuda.Event(enable_timing=True)
    # 创建模型并移至当前GPU
    model = Net().to(rank)
    # 使用FSDP封装模型并使用自定义的自动分片策略
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()  # 开始记录初始化时间
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()  # 时间记录和输出

    if rank == 0:
        init_end_event.synchronize()
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # 同步所有进程，确保训练完成
        # 避免rank 0在保存时其他进程还在运行
        dist.barrier()
        # 获取全量模型参数（会自动聚合分片）
        states = model.state_dict()
        # 仅主进程保存
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    # 销毁进程组，释放资源
    cleanup()

# 6. 修改后的参数传递和运行部分
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    fsdp_main(args) 


