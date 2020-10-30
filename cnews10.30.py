import tensorflow.keras as kr
import torch
from torch import optim
from torch import nn
from cnews_loader import read_category, read_vocab,process_file
from model import TextRNN
import numpy as np
import torch.utils.data as Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#设置数据目录
vocab_file = 'cnews.vocab.txt'
train_file = 'cnews.train1.txt'
test_file = 'cnews.test.txt'
val_file = 'cnews.val.txt'
# 获取文本的类别及其对应id的字典
categories, cat_to_id = read_category()
#print(categories)
# 获取训练文本中所有出现过的字及其所对应的id
words, word_to_id = read_vocab('cnews.vocab.txt')
#print(words)
#print(word_to_id)
#print(word_to_id)
#获取字数
vocab_size = len(words)

# 数据加载及分批
# 获取训练数据每个字的id和对应标签的one-hot形式
x_train, y_train = process_file('cnews.train1.txt', word_to_id, cat_to_id, 600)
#print('x_train=', x_train)
x_val, y_val = process_file('cnews.val.txt', word_to_id, cat_to_id, 600)


#设置GPU
cuda = torch.device('cuda')
x_train, y_train = torch.LongTensor(x_train),torch.Tensor(y_train)
x_val, y_val = torch.LongTensor(x_val),torch.Tensor(y_val)

train_dataset = Data.TensorDataset(x_train,y_train)
train_loader = Data.DataLoader(dataset = train_dataset,batch_size=1280, shuffle=True)
val_dataset= Data.TensorDataset(x_val, y_val)
val_loader = Data.DataLoader(dataset=val_dataset,batch_size=1280)

def train():
    model = TextRNN().to(device)
    #定义损失函数
    Loss = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

#保存最好模型，先给一个定义为0
    best_val_acc = 0
    costs=[]
    early_stop = 0
    min_loss = float('inf')
    for epoch in range(5):
        # print('epoch=',epoch)
        #分批训练
        losses = []
        accuracy_array0= np.array([])
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x = x_batch.to(device)
            y = y_batch.to(device)
            out = model(x)
            loss = Loss(out, y)
            losses.append(loss.item())
            #print(out)
            #print('loss=',loss)
            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy0 = np.mean((torch.argmax(out, 1)==torch.argmax(y,1)).cpu().numpy())
            accuracy_array0 = np.append(accuracy_array0, accuracy0)
        meanloss = np.mean(losses)  
        costs.append(meanloss)             
#对模型进行验证
        if  (epoch+1)% 5 == 0:  
            accuracy_train = np.mean(accuracy_array0)
            print('accuracy_train:',accuracy_train)
            for step, (x_batch,y_batch) in enumerate(val_loader):
                x = x_batch.to(device)
                y = y_batch.to(device)
                out = model(x)
                #计算准确率
            accuracy1 = np.mean((torch.argmax(out, 1)==torch.argmax(y,1)).cpu().numpy())    
            accuracy_array1= np.array([])
            accuracy_test = np.mean(accuracy_array1)
            print('accuracy_test:',accuracy_test)
            if accuracy1 > best_val_acc:
                torch.save(model,'model.pkl')
                best_val_acc = accuracy1
                print('model.pkl saved')
        #accuracy_array1 = np.append(accuracy_array1, best_val_acc
         #  早停法
        if meanloss < min_loss:
            min_loss = meanloss
            early_stop = 0
        else:
            early_stop += 1
        if early_stop > 5:
            print(f"loss连续{epoch}个epoch未降低, 停止循环")
            break
    #print('best_accuracy:',best_val_acc)     
if __name__ == '__main__':
    train()

