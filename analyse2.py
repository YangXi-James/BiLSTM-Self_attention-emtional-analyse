import math
import sys
import random

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import re
import jieba
import gensim
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F



plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
data = pd.read_csv(r'D:\360MoveData\Users\10539\Desktop\data cleaned1.csv', encoding='utf_8_sig')
# data['contents_length'] = data['contents'].apply(lambda x: len(str(x)))
# len_se = data.groupby('contents_length')['contents_length'].count()
# sns.displot(len_se, bins=100,  kde=True, rug=True)
# plt.title('游戏评论长度分布')
# plt.text(1000, 600, '评论长度的9/10分位数：134.0')
# plt.show()
# print('评论长度的9/10分位数：', data['contents_length'].quantile(0.9))
#
#
# positive = len(data['stars'][data['stars'] >= 5])
# total = len(data['stars'])
# negative = len(data['stars'][data['stars'] <=1])
# neutral = total - positive-negative
# sns.displot(data['stars'], bins=5, kde=False)
# plt.title('评论情感分布',size='10')
# plt.show()


# sns.displot(data['spent'], bins=500, kde=False)
# plt.title('游戏时长',size='10')
# plt.subplots_adjust(top=0.8)
# plt.show()

# sns.displot(Z, bins=500, kde=False)
# plt.title('支持度',size='10')
# plt.subplots_adjust(top=0.9)
# plt.show()

# num_list = data['net_support']
# plt.bar(range(len(num_list)), num_list,color='rgb')
# plt.show()
# explode = [0,0,0,]  # 生成数据，用于突出显示B
# colors=['#9999ff','#ff9999','#7777aa']  # 自定义颜色
# plt.axes(aspect='equal')
# # 绘制饼图
# plt.figure(dpi=200)
# plt.subplots_adjust(top=0.7,bottom=0.2)
# plt.pie(x = [0.362,0.304,0.344], # 绘图数据
#         explode=explode, # 突出显示B
#         labels=['正向评论','中性评论','负向评论'], # 添加教育水平标签
#         colors=colors, # 设置饼图的自定义填充色
#         autopct='%.1f%%', # 设置百分比的格式，这里保留一位小数
#         pctdistance=0.8,  # 设置百分比标签与圆心的距离
#         labeldistance = 1.1, # 设置教育水平标签与圆心的距离
#         startangle = 180, # 设置饼图的初始角度
#         radius = 2, # 设置饼图的半径
#         counterclock = False, # 是否逆时针，这里设置为顺时针方向
#         wedgeprops = {'linewidth': 1.5, 'edgecolor':'red'},# 设置饼图内外边界的属性值
#         textprops = {'fontsize':10, 'color':'black'}, # 设置文本标签的属性值
#         )
#
# # 添加图标题
# plt.title('评论情感分布')
# # 显示图形
# plt.show()


# print('正面评价: %d，占总数的%.2f%%  中性评价: %d, 占总数的%.2f%%  负面评价: %d, 占总数的%.2f%%' %
#       (positive, (positive/total*100),neutral,(neutral/total*100), negative, (negative/total*100)))

# data.dropna(subset=["contents"],inplace=True)

X = data['contents']
Y = (data['stars'])
Z=(data['net_support'])
# for i in range(0,len(Y)):
#     if 0 <= Y[i] < 2:
#         Y[i]=0
#     elif 1<=  Y[i] <4:
#         Y[i]=1
#     else :
#         Y[i]=2

for i in range(0,len(Y)):
    if 1 <= Y[i] <= 3:
        Y[i]=0
    else:
        Y[i]=1;


def drop_non_chinese(text):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese_text = re.sub(pattern, '', text)
    return chinese_text


def split_text(chinese_text, cut_all=False):
    text_generater = jieba.cut_for_search(chinese_text)
    result = ' '.join(text_generater)
    return result


def drop_stop_words(st_path, chinese_text):
    with open(st_path, 'r') as f:
        st = f.read()
        st_list = st.splitlines()
    word_list = chinese_text.split()
    for stop_word in st_list:
        word_list = [word for word in word_list if word != stop_word]
    return word_list


def train_Word2Vec_model(text, save_path):
    model = gensim.models.Word2Vec(text, vector_size=50, min_count=5, window=3)
    model.save(save_path)
    return model


def word_to_dict(X, word_vec_model):
    total_set = set()
    word_to_vec = dict()
    word_to_index = dict()

    for x in X:
        total_set = set.union(total_set, set(x))
    index = 1
    for i in total_set:
        if word_vec_model.wv.__contains__(i):
            word_to_vec[i] = word_vec_model.wv[i]
            word_to_index[i] = index
            index += 1

    return word_to_vec, word_to_index


X = X.apply(lambda x: drop_non_chinese(str(x)))
X = X.apply(lambda x: split_text(x))
X = X.apply(
     lambda x: drop_stop_words(r'D:\360MoveData\Users\10539\Desktop\emotion_analyze\stop_words.txt', x))

for i in range(0, len(X)-1):
    if Z[i] <= 0 or len(X[i]) == 0:
        X=X.drop(i)
        Y=Y.drop(i)

word_vec_model = train_Word2Vec_model(X, r'D:\360MoveData\Users\10539\Desktop\emotion_analyze\word2vec.model')
word_to_vec, word_to_index = word_to_dict(X, word_vec_model)
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=2)
print('第六条评论：', X_train[5], '       第五条评论标星：', y_train[5])


def text_to_index(X, word_to_index, max_len=30):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        j = 0
        for word in X[i][:max_len]:
            if word in word_to_index:
                X_indices[i, j] = word_to_index[word]
                j += 1

    return X_indices


def pretrained_embedding_layer(word_to_vec, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = 50
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec[word]

    weight = torch.from_numpy(emb_matrix)
    embedding = nn.Embedding.from_pretrained(weight)
    embedding.weight.requires_grad = True

    return embedding


class EmotionModel(nn.Module):
    def __init__(self, word_to_vec, word_to_index, max_len, hidden_dim):
        super(EmotionModel, self).__init__()
        self.embedding_dim = 50
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.W_Q = nn.Linear(hidden_dim*2,  hidden_dim*2, bias=False)
        self.W_K = nn.Linear(hidden_dim*2,  hidden_dim*2, bias=False)
        self.W_V = nn.Linear(hidden_dim*2,  hidden_dim*2, bias=False)
        self.embedding = pretrained_embedding_layer(word_to_vec, word_to_index)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, num_layers=5, hidden_size=self.hidden_dim, dropout=0.2,
                            batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_dim*2, 2)
        self.Logsoftmax = nn.LogSoftmax(dim=1)

    def attention(self, Q, K, V):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, V)
        output = context.sum(1)

        return output, alpha_n




    def forward(self, inputs):
        embeds = self.embedding((inputs.long()))
        lstm_out1, hidden = self.lstm(embeds.view(-1, self.max_len, self.embedding_dim).float())
        Q = self.W_Q(lstm_out1)  # [batch_size,seq_len,hidden_dim]
        K = self.W_K(lstm_out1)
        V = self.W_V(lstm_out1)
        attn_output, alpha_n = self.attention(Q, K, V)
        fc_out1 = self.linear(attn_output)
        fc_out2 = self.Logsoftmax(fc_out1)
        return fc_out2


def data_loader(X_train, y_train, batch_size=None):
    train_db = TensorDataset(torch.from_numpy(X_train).float(), torch.squeeze(torch.from_numpy(y_train)))
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=False)
    return train_loader


def train_model(X_train, y_train, X_test, y_test, word_to_vec, word_to_index, params_save_path,
                max_len=30, batch_size=50, lr=0.01, epochs=400, hidden_dim=10):
    X_train_indice = text_to_index(X_train, word_to_index, max_len=max_len)
    train_loader = data_loader(X_train_indice, y_train, batch_size=batch_size)
    model = EmotionModel(word_to_vec, word_to_index, max_len, hidden_dim)
    model = model.cuda()
    cost_func = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,amsgrad=True)
    m = len(X_train)
    num_batches = m / batch_size
    costs = []

    for epoch in range(epochs):
        epoch_cost = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            output = model.forward(batch_x)
            cost = cost_func(output, batch_y.long())
            epoch_cost += cost
            cost.backward()
            optimizer.step()

        epoch_cost /= num_batches
        costs.append(epoch_cost)
        print('Cost after epoch %i : %f' % (epoch, float(epoch_cost)))

    torch.save(model.state_dict(), params_save_path)
    print('参数已保存至本地pkl文件。')

    result=[]
    for cost in costs:
        cost=cost.cpu().detach().numpy()
        result.append(cost)
    plt.plot(result)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.show()

    model = model.cpu()
    output_train = model(torch.from_numpy(X_train_indice).float())
    pred_y_train = torch.max(output_train, dim=1)[1].data.numpy()
    print('Train Accuracy: %.2f%%' % float(np.sum(np.squeeze(y_train) == pred_y_train) / m * 100))

    X_test_indice = text_to_index(X_test, word_to_index, max_len=max_len)
    output_test = model(torch.from_numpy(X_test_indice).float())
    pred_y_test = torch.max(output_test, dim=1)[1].data.numpy()
    print('Test Accuracy: %.2f%%' % float(np.sum(np.squeeze(y_test) == pred_y_test) / len(y_test) * 100))
    return pred_y_test


def confuse_matrix(y, pred_y):
    fn = np.sum((pred_y==1)&(y==0))
    fp = np.sum((pred_y==0)&(y==1))
    tn = np.sum((pred_y==1)&(y==1))
    tp = np.sum((pred_y==0)&(y==0))
    confuse_matrix = np.array([[tp, fp], [fn, tn]])
    precision = tp / (fp+tp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision*recall) / (precision+recall)
    print(confuse_matrix)
    print('Precision = %.2f,   Recall = %.2f    F1 = %.2f' % (precision, recall, F1))
    return F1


from sklearn.naive_bayes import GaussianNB
if __name__ == '__main__':
    # X_train_indice = text_to_index(X_train, word_to_index, max_len=50)
    # X_test_indice = text_to_index(X_test, word_to_index, max_len=50)
    # mnb = GaussianNB()
    # mnb.fit(X_train_indice, y_train)
    # result=mnb.predict(X_test_indice)
    # k=confuse_matrix(result,y_test)

    max_len = 50
    batch_size = 100
    hidden_dim = 128
    epochs=[20]
    F1=[]
    params_save_path = r'D:\360MoveData\Users\10539\Desktop\emotion_analyze\emo_parms.pkl'
    for e in epochs:
        pred_y_test = train_model(X_train, y_train, X_test, y_test, word_to_vec, word_to_index, params_save_path,
                        max_len=max_len, batch_size=batch_size, hidden_dim=hidden_dim, lr=0.01, epochs=e)
        F1.append(confuse_matrix(y_test,pred_y_test))
    plt.figure()
    plt.plot(epochs,F1)
    plt.xlabel("epochs")
    plt.ylabel("F1 score")
    plt.show()

