import torch
import numpy as np
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from convo_net import CNN_v1, MyDataset
from torchvision import transforms
from autoencoder import ConvAE
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from import_images import grab_images
from generate_labels import generate_labels
from sklearn import model_selection

neural_net = CNN_v1()
neural_net.load_state_dict(torch.load('/Users/rydereasterlin/Desktop/CNN_V1.txt'))

features = grab_images('images_data.txt')
labels = generate_labels(features)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, labels, test_size=.15, random_state=0)
X_train, X_val, Y_train, Y_val =  model_selection.train_test_split(X_train, Y_train, test_size=.075, random_state=1)

X_train, X_test, X_val = X_train.reshape(23*999, 64, 64), X_test.reshape(5*999, 64, 64), X_val.reshape(2*999, 64, 64)
X_train, X_test, X_val = X_train[np.newaxis], X_test[np.newaxis], X_val[np.newaxis]
Y_train, Y_test, Y_val = Y_train.flatten(), Y_test.flatten(), Y_val.flatten()

training_set = MyDataset(X_train.transpose(1, 0, 2, 3), Y_train)
validation_set = MyDataset(X_val.transpose(1, 0, 2, 3), Y_val)
testing_set = MyDataset(X_test.transpose(1, 0, 2, 3), Y_test)

training_generator = DataLoader(training_set, batch_size=32, shuffle=True, pin_memory=True)
validation_generator = DataLoader(validation_set, batch_size=Y_val.size)
testing_generator = DataLoader(testing_set, batch_size=Y_test.size)
print('Data loaded')

autoencoder = ConvAE()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=.001, weight_decay=1e-5)
num_epochs = 200

for epoch in range(num_epochs):
    for data in training_generator:
        images, _ = data[0].float(), data[1].long()
        #img = images.view(images.size(0), -1)
        # ===================forward=====================
        output = autoencoder(images)
        loss = criterion(output, images)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)
    print(epoch)

print("Done training")
prior_counter=0
after_counter=0

autoencoder.eval()
neural_net.eval()
with torch.no_grad():
    for data in testing_generator:
        images, labels = data[0].float(), data[1].long()
        counter = 0
        for img in images:
            save_image(img, '/Users/rydereasterlin/Desktop/Rogers_Group/test_images/prior/img' + str(counter) + '.png')
            counter+=1
        outputs = autoencoder(images)
        outputs = outputs.view(4995, 1, 64, 64)
        counter = 0
        for img in outputs:
            save_image(img, '/Users/rydereasterlin/Desktop/Rogers_Group/test_images/after/img' + str(counter) + '.png')
            counter+=1
        outputs = neural_net(outputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        num_correct = 0
        for obj in c:
            if obj == 1: num_correct+=1
    acc = num_correct/Y_test.size
print("Mean accuracy on testing set was:", acc)


