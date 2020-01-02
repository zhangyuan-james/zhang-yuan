
# coding: utf-8

# In[430]:


import re
import numpy as np
import random 
changetimes=1 #允许出错次数
ppd=0.2
ppt=0.2
ppr=0.2
ppi=0.2#各类概率
leter="qwazserdxcftygvbhuijnmkopl"
def encode(c,maxlen):
    temp=[]
    for i in c:
        temp.append((ord(i)-96)*10)
    b=[0 for x in range(int(maxlen*1.2))]
    lenc=len(c)
    for i in range(len(c)):
        loc=int((i/lenc)*maxlen*1.2)
        b[loc]+=temp[i]
    return b
def decode(b):
    temp=""
    for i in b:
        if i%10>5:
            i+=5
        j=int(i/10)
        if j>0:
            temp+=chr(j+96)
    return temp
def transwords(words):
    myword=[]
    for i in words:
        temp=i.lower().strip()
        myword.append(temp)
    words=list(set(myword)) 
    myword=[]
    for i in words:
        i=re.search("[a-z]+",i)
        if i!=None:
            temp=i.group(0)
            myword.append(temp)
    myword=list(set(myword)) 
    maxlen=0
    for i in myword:
        if len(i)>maxlen:
            maxlen=len(i)
    return myword,maxlen
def deletes(a):
    num=random.randint(0,len(a)-1)
    return a[:num] + a[num+1:]
def transposes (a):
    if len(a)==1:
        return a
    else:
        num=random.randint(0,len(a)-2)
        return a[:num] +a[num+1]+a[num]+ a[num+2:]
def replaces (a):
    num=random.randint(0,len(a)-1)
    return a[:num]+misletter(a,num) + a[num+1:]
def inserts(a):
    num=random.randint(0,len(a)-1)
    return a[:num]+misletter(a,num) + a[num:]
'''
def misletter(w,num):
    return chr(random.randint(1,26)+96)
'''
def misletter(w,num):
    imin=max(leter.index(w[num])-5,0)
    #print(imin)
    imax=min(leter.index(w[num])+5,25)
    #print(imax)
    return leter[random.randint(0,imax-imin)+imin]
def makemistake(w):
    a=random.random()
    if a<=ppd:
        w=deletes(w)
        #print("deletes",w)
        #z[0]+=1
    elif a<=ppd+ppt:
        w=transposes(w)
        #print("transposes",w)
        #z[1]+=1
    elif a<=ppd+ppt+ppr:
        w=replaces(w)
        #print("replaces",w)
        #z[2]+=1
    elif a<=ppd+ppt+ppr+ppi:
        w=inserts(w)
        #print("inserts",w)
        #z[3]+=1
    else:
        w=w
        #z[4]+=1
    return w


# In[424]:


with open(r"E:\\newwords.txt",encoding = 'utf-8') as f:
    words=f.readlines()


# In[431]:


word,maxlen=transwords(words)


# In[444]:


len(word)


# In[433]:


inputsize=len(encode(word[0],maxlen))


# In[156]:


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary


# In[437]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(inputsize,390)
        self.fc2 = nn.Linear(390, 200)
        self.fc3=nn.Linear(200,inputsize)
    def forward(self, x):
        x = self.fc1(x)
        x=torch.sigmoid(x)
        x = self.fc2(x)
        x=F.relu(x)
        x = self.fc3(x)
        return x
net = Net()


# In[13]:


from tqdm import tqdm


# In[435]:



def mainfuc():
    running_loss = 0.0
    for i in range(6):
        # 获取输入
        inputs=[]
        labels=[]
        for j in word:
            temp=makemistake(j)
            inputs.append(encode(temp,maxlen))
            labels.append(encode(j,maxlen))
        inputs=torch.from_numpy(np.array(inputs))
        inputs=inputs.float()
        labels=torch.from_numpy(np.array(labels))
        labels=labels.float()
        #print("ok")
        # 梯度置0
        optimizer.zero_grad()

        # 正向传播，反向传播，优化
        outputs = net(inputs)
        
        loss =criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印状态信息
        running_loss += loss.item()
        #print(epoch + 1, i + 1, running_loss / 2000)
        losss.append([epoch, i + 1,loss])
        running_loss = 0.0


# In[438]:


losss=[]
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
try:
    with tqdm(range(2)) as t:
        for epoch in t:
            #print(epoch,end="\r")
            mainfuc()
except KeyboardInterrupt:
    t.close()
    raise
t.close()
print("搞定！")


# In[439]:


summary(net,(1,inputsize))


# In[440]:


import matplotlib.pyplot as plt
xrow=[]
yrow=[]

for i in losss:
    xrow.append(i[1]+i[0]*6)
    yrow.append(i[2])


# In[443]:


losss


# In[441]:


plt.plot(xrow,yrow)
plt.show()


# In[278]:


word[0]


# In[308]:


print(a,decode(b))


# In[309]:


encode(a,maxlen)


# In[372]:


torch.save(net,r"E:\\net.pkl")


# In[371]:


a=word[1]
b=encode(makemistake(a),maxlen)
print(a,b,decode(b))
b=torch.from_numpy(np.array(b))
b=b.float()
c=net(b)
print(c.int(),decode(c.int()))


# In[380]:


right=0
flase=0
for j in range(100):
    for i in word:
        b=encode(makemistake(i),maxlen)
        b=torch.from_numpy(np.array(b))
        b=b.float()
        c=net(b)
        if i!=decode(c.int()):
            #print(i,decode(b),decode(c.int()),c.int())
            flase+=1
        else:
            right+=1


# In[385]:



for i in word:
    right=0
    flase=0
    for j in range(100):
        b=encode(makemistake(i),maxlen)
        b=torch.from_numpy(np.array(b))
        b=b.float()
        c=net(b)
        if i!=decode(c.int()):
            #print(i,decode(b),decode(c.int()),c.int())
            flase+=1
        else:
            right+=1
    print(i,right,flase,right/(right+flase))


# In[383]:


print(right,flase,right/(right+flase))


# In[124]:


b=encode(makemistake(a),maxlen)


# In[125]:


inputs=[]
labels=[]
for j in word:
    temp=makemistake(j)
    inputs.append(encode(temp,maxlen))
    labels.append(encode(j,maxlen))


# In[67]:


inputs[90]


# In[471]:


a=torch.zeros((len(word)))
      for j in range(len(outputs)):
          b=torch.zeros((len(word)))
          for k  in range(len(word)):
              temp=criterion(outputs[j],torch.from_numpy(np.array(encode(word[k],maxlen))).float())
              b[k]+=temp
          #print(b)
          a[j]+=b.min()

