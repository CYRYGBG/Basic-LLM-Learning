import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# 1. 导入必要的库
from torch.optim.lr_scheduler import StepLR, LambdaLR
import deepspeed
import os

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

# 2. 修改train函数，主要的修改包括使用deepspeed的封装进行前向
#    和反向传播过程
def train(args, model_engine, train_loader, epoch):
    model_engine.train()
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据传递到model_engine.device自动进行设备分配
        data = data.to(model_engine.device)
        target = target.to(model_engine.device)
        # 使用model_engine.device进行反向传播相关处理
        # 进行修改的包括：梯度清零, 前向传播，反向传播和梯度更新
        model_engine.zero_grad()  # 原 optimizer.zero_grad()
        output = model_engine(data)  # 原 output = model(data)
        loss = F.nll_loss(output, target)
        model_engine.backward(loss)  # 原 loss.backward()
        model_engine.step()  # 原 optimizer.step()

        if batch_idx % args.log_interval == 0 and local_rank == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 所有损失值求和
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测标签
            correct += pred.eq(target.view_as(pred)).sum().item() # 预测正确的样本数

    test_loss /= len(test_loader.dataset)

    if local_rank == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main(args):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    
    # 定义好优化器和调度器之后再传递给deepspeed.initialize进行接管
    # 这里要不就完全自己指定在传递给deepspeed，要不就完全在config中指定好，不能进行混用
    # 另外，在代码中指定的话会覆盖config中的设置
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # default=1.0

    # 计算每个epoch的batch数量
    total_batches_per_epoch = len(train_loader)

    # 定义基于batch的调度逻辑
    def lr_lambda(current_step):
        # 每隔 total_batches_per_epoch * step_size 个batch更新一次
        return 1.0 if current_step % (total_batches_per_epoch * args.step_size) !=0 else args.gamma
    scheduler = LambdaLR(optimizer, lr_lambda)  

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # default=0.7
    # 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
                                                        args=args,
                                                        model=model,
                                                        model_parameters=model.parameters(),
                                                        optimizer=optimizer,
                                                        lr_scheduler=scheduler,
                                                        config=args.deepspeed_config
                                                        )

    
    for epoch in range(1, args.epochs + 1):
        train(args, model_engine, train_loader, epoch)
        # 测试函数传递的是当前的模型
        test(model_engine.module, model_engine.device, test_loader)
        # 不需要再进行显式调用，会在model_engine.step()中进行统一处理
        # scheduler.step()

    if args.save_model:
        torch.save(model_engine.module.state_dict(), "mnist_cnn.pt")


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
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--step-size', type=int, default=1, 
                        help='Enable DeepSpeed')  
    parser.add_argument('--deepspeed', action='store_true', default=True, 
                        help='Enable DeepSpeed')
    parser.add_argument('--deepspeed_config', default='ds_config.json',
                        help='DeepSpeed config file path')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # 运行命令`deepspeed --num_gpus=2 deepspeed_main.py --deepspeed_config ds_config.json`
    main(args)
