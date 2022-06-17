import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


BATCH_SIZE = 32
ecg_train = np.load('train/ecg_train.npy')
ecg_test = np.load('train/ecg_test.npy')
train_labels = np.load('train/train_labels.npy')
test_labels = np.load('train/test_labels.npy')
ecg_train = np.expand_dims(ecg_train, axis=1)
ecg_test = np.expand_dims(ecg_test, axis=1)
print('Size of training data: {}'.format(ecg_train.shape))
print('Size of testing data: {}'.format(ecg_test.shape))
class Dataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

train_X,test_X, train_y, test_y = train_test_split(ecg_train,train_labels,test_size = 0.2,random_state = 42)

train_set = Dataset(train_X,  train_y)
val_set = Dataset(test_X, test_y)
test_set = Dataset(ecg_test, test_labels)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


drop = 0.3
class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()

        self.cnnlayer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=32 ,padding = 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=drop)
        )
        self.cnnlayer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32 ,padding = 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=drop)
        )

        self.cnnlayer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32 ,padding = 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=drop)
        )
        self.cnnlayer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32 ,padding = 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=drop)
        )
        self.cnnlayer5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32 ,padding = 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=drop)
        )
        self.cnnlayer6 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32 ,padding = 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=drop)
        )
        self.cnnlayer7 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32 ,padding = 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=drop)
        )
        self.cnnlayer8 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32 ,padding = 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=drop)
        )
        self.cnnlayer9 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32 ,padding = 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=drop)
        )
        self.cnnlayer10 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32 ,padding = 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=drop)
        )

    def forward(self, x):
        x = self.cnnlayer1(x)
        x = self.cnnlayer2(x)
        x = self.cnnlayer3(x)
        x = self.cnnlayer4(x)
        x = self.cnnlayer5(x)
        x = self.cnnlayer6(x)
        x = self.cnnlayer7(x)
        x = self.cnnlayer8(x)
        x = self.cnnlayer9(x)
        x = self.cnnlayer10(x)

        return x
class Transformer(nn.Module):
    def __init__(self, ):
        super(Transformer, self).__init__()
        self.precnnlayer = Model()
        encoder_layers = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=128,
                                                    dropout=0.30)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=2,
                                                         norm=nn.LayerNorm(64))

        self.pred_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.precnnlayer(x)
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = F.avg_pool1d(x, kernel_size=x.size()[2])
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.pred_layer(x)
        return x
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
device = get_device()
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# fix random seed for reproducibility
same_seeds(42)
# training parameters
num_epoch = 70  # number of training epoch
learning_rate = 0.001  # learning rate
# the path where checkpoint saved
model_path = './1minbestmodel.ckpt'
# create model, define a loss function, and optimizer
model = Transformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-8)
best = 0.0
best_test = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        #         print(inputs.shape)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer.step()
        #         print(train_pred.cpu().sum(),labels.cpu().sum())
        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    predict = []
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            _, val_pred = torch.max(outputs, 1)

            for m in val_pred.cpu().numpy():
                predict.append(m)
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item()# get the index of the class with the highest probability
            val_loss += batch_loss.item()
        #
        # train_loss_list.append(train_loss/ len(train_loader))
        # train_acc_list.append(train_acc / len(train_set))
        # val_loss_list.append(val_loss/ len(val_loader))
        # val_acc_list.append(val_acc / len(val_set))

        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
            val_acc / len(val_set), val_loss / len(val_loader)
        ))

        predict = np.array(predict)
        predict_val= predict.reshape(predict.shape[0], 1)
        labelsval = test_y.reshape(test_y.shape[0], 1)
        cm = confusion_matrix(labelsval, predict_val)
        acc = (cm[1,1]+cm[0,0])/(cm[1,1]+cm[1,0]+cm[0,0]+cm[0,1])
        sen = cm[1,1]/(cm[1,1]+cm[1,0])
        spe = cm[0,0]/(cm[0,0]+cm[0,1])
        pre = cm[1,1]/(cm[1,1]+cm[0,1])
        avg = (acc+sen+pre)/3
        print('VAL:acc= {:.3f} sen = {:.3f} spe = {:.3f} pre = {:.3f} avg = {:.3f}'.format(acc,sen,spe,pre,avg))
        # if the model improves, save a checkpoint at this epoch
        
        if avg > best:
            best = avg
            torch.save(model.state_dict(), model_path)
            print('save model',cm)
            # print('acc= {:.3f} sen = {:.3f} spe = {:.3f}'.format(acc,spe,sen))

    model.load_state_dict(torch.load(model_path))
    test_predict = []
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, test_pred = torch.max(outputs, 1)

            for y in test_pred.cpu().numpy():
                test_predict.append(y)
    test_predict = np.array(test_predict)
    predict_test = test_predict.reshape(test_predict.shape[0], 1)
    labels_test = test_labels.reshape(test_labels.shape[0], 1)
    test_cm = confusion_matrix(labels_test, predict_test)
    test_acc = (test_cm[1, 1] + test_cm[0, 0]) / (test_cm[1, 1] + test_cm[1, 0] + test_cm[0, 0] + test_cm[0, 1])
    test_sen = test_cm[1, 1] / (test_cm[1, 1] + test_cm[1, 0])
    test_spe = test_cm[0, 0] / (test_cm[0, 0] + test_cm[0, 1])
    test_pre = test_cm[1, 1] / (test_cm[1, 1] + test_cm[0, 1])
    test_avg = (test_acc  + test_sen + test_pre) / 3
    print('TESTacc= {:.3f} sen = {:.3f} spe = {:.3f} pre = {:.3f} avg = {:.3f}'.format(test_acc, test_sen,test_spe, test_pre,test_avg))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')
