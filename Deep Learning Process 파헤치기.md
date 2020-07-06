# Deep Learning Process 파헤치기

> 이 글은 kaggle 문제를 풀어 보면서 느낀 팁들과 데이터 분석에 기본적인 코드 흐름을 분석하여 기술한 내용이다.



## 1.데이터 수집하기

> 당연한 말이지만 머신 러닝 모델을 구현하기 위해선 가장 먼저 데이터를 수집해야 한다.



## 2.컬럼의 의미 꿰뚫기



> 수집한 데이터의 각 컬럼의 뜻과 각 컬럼의 내제적 의미와 상호 연관성을 찾아내야 한다. 
>
> 데이터를 분석하기 위해서는 대부분 직관력으로 찾아 낼 수 있다.
>
> 예를 들면 타이타닉 호에서 나이가 어린 사람이 많이 생존 했을까?  아니면 나이가 많은 사람이 많이 생존 했을까?  당신은 나이가 어린 사람이 더 많이 생존 했을 것이라 생각 했을 것이다. 
>
> 이 과정이 결국 직관을 이용한 데이터 분석이다. 
>
> 그런데 만약 데이터를 분석하기 위한 직관력이 부족하거나 직관적으로 찾기 어려운 데이터를 만났다면 다른 방법이 없을까? 
>
> 그 해결책은 간단하다. 그래프를 그려보면 된다. 
>
> python에는 그래프를 그리는데 도움을 주는 툴인 matplotlib가 존재하다
>
> 



## 	특징 추출하기

* 각 컬럼의 의미를 꿰뚫고 그 컬럼들의 특징들을 추출하는 이 두 과정을(이름의  Mr.  ,Mrs.  정보를 통해 여자와 남자 특징을 추출할 수 있다 ) 유식한 말로 feature engineering이라고 한다..



## 4.모델링

>  모든 feature를 추출했다면 이제 남은 일은 인공지능 모델링을 한다.  
>
> 아래의 코드는 pytorch를 이용하여 모델링을 하였다.

```python
def model(nn.Model):
	def __init__(self,x):
		super.__init__(model,self)
		self.fc1 = nn.Linear(100,200)
		self.fc2 = nn.Linear(200,300)
		self.fc3 = nn.Linear(300,200)	
		self.fc4 = nn.Linear(200,100)
		self.fc5 = nn.Linear(100, 10)
		self.dropout = nn.Dropout(p=0.5)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
```



> 모델을 구현 했으면 feature engineering을 거친 train data들을 모델에 넣어서 원하는 결과값이 나오도록 W(가중치),b(편향)의 값을 조정한다.
>
> 각 feature를 모델에 넣어서 훈련 시키는 코드를 아래에 적었다. 그 흐름을 직관적으로 이해하면 된다.

```python
or epoch in range(n_epochs):
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(X_train[start:end]))
        y_var = Variable(torch.FloatTensor(y_train[start:end]).squeeze())

        optimizer.zero_grad()
        output = model(x_var)
        loss = torch.sqrt(criterion(torch.log(output.squeeze()), torch.log(y_var)))
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        X_val = Variable(torch.FloatTensor(X_val))
        y_val = Variable(torch.FloatTensor(y_val))
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val.squeeze())

        if val_loss < val_loss_min:
            print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(val_loss_min, val_loss))
            torch.save(model.state_dict(), "./output/model.pt")
            val_loss_min = val_loss

        print('')
        print("Epoch: {} \tValidation Loss: {}".format(epoch+1, val_loss))

```

## 요약

1. 데이터 수집
2. 컬럼의 의미 찾기
3. feature 추출하기
4. modeling 하기
5. trainning 하기
6. 결과 도출