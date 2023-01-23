import os
import math
import numpy as np
import pandas as pd

# import torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets
from torchvision.utils import make_grid

# import ml network
from model.model import Net

# Machine Learning Parameters
BATCH_SIZE = 16				# Batch of images per forward pass
LR = 0.0001					# Learning Rate using Cross Entropy Loss
EPOCHS = 50 			    # Training Cycles over dataset
classes = ['adenomatous', 'hyperplastic', 'unspecified']

# Using SummaryBoard Writer for TensorBoard
#writer = SummaryWriter(f'runs/tensorboard')

def softmax(x, X):
	e_sum = 0.0
	for j in X:
		e_sum += math.exp(j)
	return math.exp(x) / e_sum


def train(epochs, loader, model, optimizer, criterion, hardware):
    # Configure for hardware
    model.to(hardware)

    train_result = []
    epoch_result = []

    # step for loss update
    step = 0

    epoch = 0

    # loop over the dataset multiple times
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # training loop
        model.train()
        for batch_idx, (feature, label) in enumerate(loader['train']):
            feature, label = feature.to(hardware), label.to(hardware)

            optimizer.zero_grad()

            log_ps, aux_output = model(feature)
            loss1 = criterion(log_ps, label)
            loss2 = criterion(aux_output, label)
            loss = loss1 + 0.4 * loss2

            loss.backward()
            optimizer.step()

            train_loss += (1 / (batch_idx + 1)) * (loss.item() - train_loss)

            # update tensorboard with training loss for each batch
            #writer.add_scalar('Training Loss', loss, global_step=step)
            step += 1
        epoch += 1

        if epoch % 10 == 0:
            model.eval()
            torch.save(model.state_dict(), 'model/polyp_classification_networkV' + str(epoch) +'.pt')
            print('Saving Model...')
            model.train()
    
    # end model training for batch
    model.eval()
        
    print('Epoch: ' + str(epoch + 1) + ' Training Loss: ' + str(train_loss))

    return model


def test(loaders, model, criterion, hardware):
    test_loss = 0.0
    correct = 0
    total = 0

    labels = []
    adenomatous = []
    hyperplastic = []
    unspecified = []

    # step for loss
    step = 0

    # Moving model to gpu
    model.to(hardware)

    # test loop
    model.eval()
    
    for batch_idx, (feature, label) in enumerate(loaders['test']):
        feature, label = feature.to(hardware), label.to(hardware)

        log_ps = model(feature)
        loss = criterion(log_ps, label)

        test_loss += (1 / (batch_idx + 1)) * (loss.item() - test_loss)

        pred = log_ps.data.max(1, keepdim=True)[1]
        corr = np.sum((label.t()[0] == pred).cpu().numpy())
        correct += corr
        total += label.shape[0]

        # update tensorboard with testing loss 
        # running accuray and image grid, finally display histogram of last fully connect layer
        img = feature.reshape(feature.shape[0], -1)
        class_labels = [classes[i] for i in label]
        img_grid = make_grid(feature)
        #writer.add_image('ISIC_images', img_grid)
        #writer.add_scalar('Testing Loss', loss, global_step=step)
        #writer.add_scalar('Testing Accuracy', corr / label.shape[0], global_step=step)
        #writer.add_histogram('fc', model.fc[6].weight)
        #writer.add_embedding(img, metadata=class_labels, label_img=feature, global_step=step)
        step += 1

        list_preds = log_ps.data.cpu().tolist()
        list_label = label.cpu().tolist()
        for idx, pred in enumerate(list_preds):
            labels.append(list_label[idx])
            adenomatous.append(softmax(pred[0], pred))
            hyperplastic.append(softmax(pred[1], pred))
            unspecified.append(softmax(pred[2], pred))

    # akiec bcc bkl df mel nv vasc 
    dic = {
        'label' : labels,
        'adenomatous' : adenomatous,
        'hyperplastic' : hyperplastic,
        'unspecified': unspecified
    }

    df = pd.DataFrame(dic)
    df.to_csv('results.csv', index=True)

    print('Test Loss: ' + str(test_loss))
    print('Test Accuracy: ' + str(100*correct/total) + ' ' + str(correct) + '/' + str(total))


def main():
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  
    ])

    # InceptionV3
    #	-> Image Size 299 x 299
    #	-> Images Must bes Normalized
    # 	-> Apply Image transformation to prevent model from over fitting.
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    training_folder = os.path.join(os.getcwd(), 'dataset/training')
    testing_folder = os.path.join(os.getcwd(), 'dataset/testing')
    train_dataset = datasets.ImageFolder(training_folder, transform=train_transform)
    test_dataset = datasets.ImageFolder(testing_folder, transform=test_transform)
    
    print('Loaded Dataset')
    print('Length Training dataset: ' + str(len(train_dataset)))
    print('Length Testing dataset: ' + str(len(test_dataset)))

    # load testing data into batches
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    loaders = {'train': train_loader, 'test': test_loader}


    # Optimizer and Loss definition
    model_criterion = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(Net.parameters(), LR)

    # checking for gpu
    hardware = ''
    if torch.cuda.is_available():
        hardware = 'cuda'
    else:
        hardware = 'cpu'

    # begin training loop
    model = train(EPOCHS, loaders, Net, model_optimizer, model_criterion, hardware)
    print('Training Complete!')

    test(loaders, Net, model_criterion, hardware)

    # save model
    torch.save(model.state_dict(), 'model/polyp_classification_network.pt')
    print('Saving Model...')

if __name__ == '__main__':
    main()
