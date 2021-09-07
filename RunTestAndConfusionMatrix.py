import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

import pandas as pd

def main():
    log, logclose = create_logger('test.log')

    load_model_dir = '/content/pretrained/'
    load_model_name = '240_12push0.8884.pth' 
    load_model_path = os.path.join(load_model_dir, load_model_name)

    log('load model from ' + load_model_path)
    test_dir = '/content/dataset/cdsv5/test/'
    
    #load the model
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    #end loading model

    normalize = transforms.Normalize(mean=mean, std=std)

    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape

    class_specific = True
    
    #if test

    test_batch_size = 2
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    log('test set size: {0}'.format(len(test_loader.dataset)))

    #accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
    #                class_specific=class_specific, log=print)

    #print("done. accuracy:" + str(accu))

    #end if test

    logclose()

    
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    #CONFUSION GENERATIOn
    def get_all_preds(model, dataloader):
        all_preds = torch.tensor([])
        all_preds = all_preds.cuda()
        all_labels = torch.tensor([])
        all_labels = all_labels.cuda()

        for i, (image, label) in enumerate(dataloader):
            input = image.cuda()
            target = label.cuda()

            with torch.no_grad():
                output, min_distances = model(input)
            
            all_preds = torch.cat(
                (all_preds, output)
                ,dim=0
            )
            all_labels = torch.cat(
                (all_labels, target)
                ,dim=0
            )

        return all_preds, all_labels
        
    def get_name_wronglyclassified(preds, labels, dataset):
        incorrect_predctions = {}
        incorrect_predctions["image name"]  = []
        incorrect_predctions["correct class"]  = []
        incorrect_predctions["classified class"]  = []

        pred_classes = preds.argmax(dim=1)
        for i, pred in enumerate(pred_classes):
            if (pred != labels[i]):
                incorrect_predctions["image name"].append(dataset.samples[i][0])
                incorrect_predctions["correct class"].append(int(dataset.samples[i][1]))
                incorrect_predctions["classified class"].append(pred.item())
               
        return incorrect_predctions
    
    def printIncorrectPreds(incorrectObj):
        logfileName = 'incorrect_classifications.csv'
        if os.path.exists(logfileName):
            os.remove(logfileName)
        log, logclose = create_logger(logfileName)
        log("image name,correct class,classified class")
        for i in range(0, len(incorrectObj["image name"])):
            log(incorrectObj["image name"][i] + "," + str(incorrectObj["correct class"][i]) + "," + str(incorrectObj["classified class"][i]))
        logclose()

    train_preds, train_labels = get_all_preds(ppnet_multi, test_loader)
    preds_correct = get_num_correct(train_preds, train_labels)
    incorrect = get_name_wronglyclassified(train_preds, train_labels, test_dataset)
    printIncorrect = True
    if (printIncorrect):
        printIncorrectPreds(incorrect)
    
    print('total correct:', preds_correct)
    print('accuracy:', (preds_correct / len(test_dataset)) * 100 )

    stacked = torch.stack(
            (
                train_labels
                ,train_preds.argmax(dim=1)
            )
            ,dim=1
        )

    cmt = torch.zeros(5,5, dtype=torch.int32) #5 is the number of classes
    for p in stacked:
        tl, pl = p.tolist()
        tl = int(tl)
        pl = int(pl)
        cmt[tl, pl] = cmt[tl, pl] + 1

    #Plot CM
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from plotcm import plot_confusion_matrix
    
    cm = confusion_matrix(train_labels.cpu(), train_preds.argmax(dim=1).cpu())
    print("Confusion Matrix:")
    print(cm)

    def numclasses2string(numclasses):
        d = {"0": "Bacterial blight", "1": "Brown streak", "2": "Green mottle", "3": "Mosaic", "4": "Healthy"}
        C = (pd.Series(numclasses)).map(d) #convert the list to a pandas series temporarily before mapping
        return list(C)

    plt.figure(figsize=(5,5))
    plot_confusion_matrix(cm, numclasses2string(test_dataset.classes),
     True, 'Confusion matrix', cmap=plt.cm.Greens)
    plt.savefig('confusion_matrix.eps', format='eps')
    plt.savefig('confusion_matrix.png', format='png')
    plt.show()

if __name__ == "__main__":
    main()
