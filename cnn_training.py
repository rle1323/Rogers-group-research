import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms
import time
from import_images import grab_images
from generate_labels import generate_labels
from sklearn import model_selection
from convo_net import MyDataset, CNN_v1, DenseNet
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import confusion_matrix
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

features = grab_images('images_data.txt')
labels = generate_labels(features)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, labels, test_size=.15, random_state=0)
X_train, X_val, Y_train, Y_val =  model_selection.train_test_split(X_train, Y_train, test_size=.075, random_state=1)

X_train, X_test, X_val = X_train.reshape(23*999, 64, 64), X_test.reshape(5*999, 64, 64), X_val.reshape(2*999, 64, 64)
X_train, X_test, X_val = X_train[np.newaxis], X_test[np.newaxis], X_val[np.newaxis]

Y_train, Y_test, Y_val = Y_train.flatten(), Y_test.flatten(), Y_val.flatten()
'''
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, labels, test_size=0.9, random_state=1)
print(X_train.shape)
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_train, Y_train, test_size=0.1, random_state=1)
X_train, X_test, X_val = X_train[np.newaxis], X_test[np.newaxis], X_val[np.newaxis]
'''

training_transformations = transforms.Compose([
    transforms.ToPILImage(),

    transforms.RandomHorizontalFlip(p=.5),
    transforms.RandomResizedCrop(64, scale=(.1,.75)),
    transforms.RandomRotation(45),
    transforms.RandomAffine(45),
    transforms.RandomErasing(p=.5, scale=(.05,.33)),
    transforms.RandomVerticalFlip(p=.5),

    transforms.ToTensor()
    ])
base_training_set = MyDataset(X_train.transpose(1, 0, 2, 3), Y_train)
aug_training_set = MyDataset(X_train.transpose(1, 0, 2, 3), Y_train, training_transformations)

training_set = ConcatDataset([base_training_set, aug_training_set])
validation_set = MyDataset(X_val.transpose(1, 0, 2, 3), Y_val)
testing_set = MyDataset(X_test.transpose(1, 0, 2, 3), Y_test)

training_generator = DataLoader(training_set, batch_size=256, shuffle=True, pin_memory=True)
validation_generator = DataLoader(validation_set, batch_size=Y_val.size)
testing_generator = DataLoader(testing_set, batch_size=Y_test.size)

print("Data loaded")


neural_net = CNN_v1()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(neural_net.parameters(), lr=0.001)

epoch = 0
validation_loss = []
training_loss = []
loss_change = []
validation_accuracies = []
term = ''
while epoch < 200: #and not neural_net.converged(loss_change): # loop over the dataset multiple times
    neural_net.train()
    for inputs, labels in training_generator:
        neural_net.train()
        outputs = neural_net(inputs.float())
        # zero the parameter gradients, backward + optimize
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    training_loss.append(loss)
        

    with torch.no_grad():
        neural_net.eval()
        for data in validation_generator:
            inputs, labels = data[0].float(), data[1].long()
            outputs = neural_net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            total_labels = labels.shape[0]
            num_correct = 0
            for obj in c:
                if obj == 1: num_correct+=1
            acc = num_correct/total_labels
            validation_accuracies.append(acc)
            print(acc)
            loss = criterion(outputs, labels).item()
            validation_loss.append(loss)
            if epoch > 0: 
                change = validation_loss[epoch-1] - loss
                loss_change.append(change)
            ''' 
            if acc > .7:
                term = input("End training?")
                '''
                
        epoch+=1

print('Epochs Trained:', epoch)

'''
neural_net = CNN_v1()
neural_net.load_state_dict(torch.load('/Users/rydereasterlin/Desktop/CNN_V1.txt'))
neural_net.eval()
'''
#Testing net
accuracies = []
for i in range(100):
    total_labels = Y_test.shape[0]
    num_correct = 0
    with torch.no_grad():
        for data in testing_generator:
            images, labels = data[0].float(), data[1].long()
            outputs = neural_net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for obj in c:
                if obj == 1: num_correct+=1
            if i == 99: 
                try:
                    print(confusion_matrix(labels, predicted))
                except:
                    pass
    acc = num_correct/total_labels
    accuracies.append(acc)
mean_acc = sum(accuracies)/len(accuracies)
print("Mean accuracy on testing set was:", mean_acc)

prompt = input("Save this model?")
if prompt == 'yes': torch.save(neural_net.state_dict(), '/Users/rydereasterlin/Desktop/CNN_V1.txt')


validation_loss = np.asarray(validation_loss)
training_loss = np.asarray(training_loss)

class_names = ['0-50mPa', '50-100mPa', '100-150mPa', '150-200mPa', '200-250mPa', '250-300mPa', '300-350mPa', '350-400mPa', '400-450mPa', '450-500mPa']


#plt.savefig("/Users/rydereasterlin/Desktop/Rogers_Group/conf_matrix.png")
#plt.legend(loc='upper right', shadow=True, fontsize='medium')



