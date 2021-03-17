손실함수를 구할 때보면 torch.nn.CrossEntropyLoss와  torch.nn.NLLLoss 두 함수를 만나 볼 수 있다. 둘다 cross-entropy 손실을 구하는 함수이고 마찬가지로 분류 문제처럼 출력이 확률값일 때 사용된다.
 
 - torch.nn.CrossEntropyLoss를 사용하는 예
 ```python
 for epcoh in range(epochs + 1):
    avg_cost = 0
    for x, y in data_load:
        x = x.to(device)
        y = y.to(device)
        h = model(x)
        cost = criterion(h, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / batch

    print('Epoch : {:4d}/{} Cost : {:.6f}'.format(epcoh, epochs, avg_cost))

 ```

 - torch.nn.NLLLoss를 사용하는 예
 ```python
 for batch_index, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()

        if phase== 'training':
            optimizer.zero_grad()

        output = model(data)
        print(output, target)
        loss = nn.NLLLoss(output, target)

        if phase== 'training':
            loss.backward()
            optimizer.step()
        if phase== 'validation':
            exp_lr_scheduler.step()
        
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)

    print ('[{}] epoch: {:2d} loss: {:.8f} accuracy: {:.8f}'.format(phase, epoch, loss, accuracy))
 ```

 얼핏 보면 똑같은 것 같지만 다른 면모가 있다. 결론적으로는 CrossEntropyLoss가 더 많은 일을 한다. CrossEntropyLoss 안에는 LogSoftmax + NLLLoss가 함께 사용된다. 결국 CrossEntropyLoss는 SoftMax를 적용하고 손실 값을 구하게 된다. 그렇다면 CrossEntropyLoss를 적용한 모델은 모델 자체에 Softmax가 없을 것이고 NLLLoss만 적용한 모델은 모델 마지막 레이어에 Softmax가 있을 것이다. 코드로 살펴보고 글을 마무리 해보자. 

 - CrossEntropyLoss가 적용된 모델  
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1, stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), padding=1, stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=0, stride=(2, 2))
        )

        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
```
forward 코드르 보면 선형 레이어를 거쳐 그대로 나오는 것을 확인할 수 있다.


 - NLLLoss가 적용된 모델 
 ```python
 class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2,2)))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
```
마지막 선형 레이어를(fc2) 지나 softmax 함수가 적용되는 것을 확인할 수 있다.
