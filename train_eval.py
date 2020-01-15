import torch
import torch.nn as nn
import time
import torch.nn.functional as F

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()                       #train mode
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)             #交叉熵损失函数
            loss.backward()                                     #bp计算梯度
            optimizer.step()                                    #update weights

