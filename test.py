import os
import pandas as pd
import sys
import importlib
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from mydataset import CustomDatasetFromImages
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import imp
import logging
from path import Path
from densesharp import mynet

testcsv_path="./sampleSubmission.csv" #直接指向csv文件
testdata_path='./test/'  #指向xx.npz所在文件夹
traincsv_path="./train.csv"#直接指向csv文件
valcsv_path="./val.csv"#直接指向csv文件
trainvaldata_path="./train_val/" #指向xx.npz所在文件夹
batch_size_train=64
batch_size_val=32
LR=1.e-5
EPOCH=300

def resize(voxel, seg):#Keep only the part with value and expand to 32 , and Regularize
    batch= voxel.shape[0]
    voxel=voxel*seg
    result = torch.zeros(batch, 32, 32, 32)
    voxel=torch.from_numpy(voxel)
    voxel=voxel.to(dtype=torch.float32)
    seg=torch.from_numpy(seg)
    seg=seg.to(dtype=torch.bool)#*
    x = seg.sum(dim=2).sum(dim=2)#H and W ,compress to C ,find the value
    y = seg.sum(dim=1).sum(dim=2)#C and W ,compress to H ,find the value
    z = seg.sum(dim=1).sum(dim=1)#C and H ,compress to W ,find the value
    transform =transforms.Compose([transforms.ToPILImage(),transforms.Resize([32,32]),transforms.ToTensor()])
    for i in range(batch):
        xs = x[i, :].nonzero()
        ys = y[i, :].nonzero()
        zs = z[i, :].nonzero()
        #print(xs)
        cropped = voxel[i, xs[0]:xs[-1], ys[0]:ys[-1], zs[0]:zs[-1]]
        sizex, sizey, sizez = cropped.shape
        tmp = torch.zeros(sizex, 32, 32)
        for j in range(sizex):
            tmp[j] = transform(cropped[j])
        for j in range(32):
            result[i, :, j, :] = transform(tmp[:, j, :])
        std, mean = torch.std_mean(result[i])
        result[i] = (result[i] - mean) / std#改之前
    return result

def main(args):
    if args.mode=='train':
        train_model()
    if args.mode=='test':
        test_model()

def train_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data=pd.read_csv(traincsv_path)
    train_label=np.zeros(393)
    print('read train images')
    for i in range(393):
        print(i+1,'/393')
        npz=np.load(trainvaldata_path+str(data.loc[i,'name'])+'.npz')
        train_label[i]=data.loc[i,'lable']
        if i==0 :
            voxel=np.expand_dims(npz['voxel'], axis=0)
            seg= np.expand_dims(npz['seg'], axis=0)#1,100,100,100
        else:
            voxel = np.append(voxel, np.expand_dims(npz['voxel'], axis=0), axis=0)
            seg = np.append(seg, np.expand_dims(npz['seg'], axis=0), axis=0)#x,100,100,100
    print('resize train images')
    prepare_train=resize(voxel,seg)
    prepare_train=prepare_train.unsqueeze(1)
    train_label = torch.from_numpy(train_label)
    train_label = train_label.to(dtype=torch.float32, device=device)
    dset_train=TensorDataset(prepare_train, train_label)
    print('completed')
    print('read val images')
    data=pd.read_csv(valcsv_path)
    val_label=np.zeros(72)
    for i in range(72):
        print(i+1,'/72')
        npz=np.load(trainvaldata_path+str(data.loc[i,'name'])+'.npz')
        val_label[i]=data.loc[i,'lable']
        if i==0 :
            voxel_val=np.expand_dims(npz['voxel'], axis=0)
            seg_val= np.expand_dims(npz['seg'], axis=0)#1,100,100,100
        else:
            voxel_val = np.append(voxel_val, np.expand_dims(npz['voxel'], axis=0), axis=0)
            seg_val = np.append(seg_val, np.expand_dims(npz['seg'], axis=0), axis=0)#x,100,100,100
    print('resize val images')
    prepare_val=resize(voxel_val,seg_val)
    prepare_val=prepare_val.unsqueeze(1)
    val_label = torch.from_numpy(val_label)
    val_label = val_label.to(dtype=torch.float32, device=device)
    dset_val=TensorDataset(prepare_val, val_label)
    print('completed')
    
    train_loader = DataLoader(dset_train, batch_size_train, shuffle=True)
    train_loader_mix = DataLoader(dset_train, batch_size_train, shuffle=True)
    val_loader = DataLoader(dset_val, batch_size_val, shuffle=True)
    
    print('start train')
    model=mynet().cuda()
    model.load_state_dict(torch.load('./best_in_mynet.pkl'))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 1.1)
    max_acc = 0
    best_loss=1.
    kk=1.414
    kkk=1.414
    for epoch in range(EPOCH): 
        scheduler.step()
        
        model.train()
        if(epoch<100):
            alpha=0.8
        if(epoch>100 and epoch<150):
            alpha=0.4
        if(epoch>150):
            alpha=1.2
        j=-1;
        print('training ',epoch)
        if(epoch==25 or epoch==50 or epoch==75 or epoch==100 or epoch==125 or epoch==150 or epoch==175 or epoch==200 or epoch==225 or epoch==250 or epoch==275 or epoch==300):
            train_loader = DataLoader(dset_train, batch_size_train, shuffle=True)
            train_loader_mix = DataLoader(dset_train, batch_size_train, shuffle=True)
            print('update data')
        total = torch.Tensor([0])
        correct = torch.Tensor([0])
        total_loss = 0.
        n=0
        for  datax,datay in zip(train_loader,train_loader_mix):
            j=j+1;
            voxel,label=datax
            voxel_m,label_m=datay
            lam=np.random.beta(alpha,alpha)
            #print(lam)
            voxel=lam*voxel+(1-lam)*voxel_m
            voxel = voxel.cuda()
            label = label.cuda()
            label_m = label_m.cuda()
            prediction = model(voxel)
            if(epoch==0):
                loss = lam*loss_func(prediction, label.to(dtype=torch.float32))+(1-lam)*loss_func(prediction, label_m.to(dtype=torch.float32))
            else:
                loss = lam*loss_func(prediction, label.to(dtype=torch.float32))+(1-lam)*loss_func(prediction, label_m.to(dtype=torch.float32))+avg_val_loss*kk*kkk
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n+=1
            right_label=torch.round(lam*label.to(dtype=torch.float32)+(1-lam)*label_m.to(dtype=torch.float32))
            #_, predicted = torch.max(prediction.detach(), 1)
            predicted =prediction.detach()
            total += label.size(0)
            predicted[predicted > 0.5] = 1
            predicted[predicted < 0.5] = 0
            correct += (predicted == right_label.round()).cpu().sum()
        avg_train_acc =correct.item() / total.item()
        avg_train_loss = total_loss / n
        if(avg_train_acc>0.86):
            kk=10
        else:
            kk=1.414 
        print('train loss: ', avg_train_loss)
        print('train accuracy: ', avg_train_acc)
        
        model.eval()
        total = torch.Tensor([0])
        correct = torch.Tensor([0])
        total_loss = 0.0
        n = 0
        for j, (voxel, label) in enumerate(val_loader):
            voxel = voxel.cuda()
            label = label.cuda()
            prediction = model(voxel)
            loss = loss_func(prediction, label.to(dtype=torch.float32))
            #_, predicted = torch.max(prediction.detach(), 1)
            predicted =prediction.detach()
            n+=1
            total_loss += loss.item()
            total += label.size(0)
            predicted[predicted > 0.5] = 1
            predicted[predicted < 0.5] = 0
            correct += (predicted == label.round()).cpu().sum()
        avg_val_acc =correct.item() / total.item()
        avg_val_loss = total_loss / n
        if(avg_val_acc>0.69):
            optimizer = torch.optim.Adam(model.parameters(), lr=1.e-6)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        if(avg_val_acc<0.542):
            kkk=10
        else:
            kkk=1.414 
        if(avg_val_acc>0.626):
            kk=1.414
        print('val loss: ', avg_val_loss)
        print('val accuracy: ', avg_val_acc)
        
        if avg_val_acc > max_acc:
            max_acc = avg_val_acc
            torch.save(model.state_dict(), 'best_acc_display.pkl')
        print('best accuracy:',max_acc)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_loss_display.pkl')
        print('best loss:',best_loss)
    print('finished')


def test_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data=pd.read_csv(testcsv_path)

    print('read images')
    for i in range(117):
        npz=np.load(testdata_path+str(data.loc[i,'Id'])+'.npz')
        if i==0 :
            voxel=np.expand_dims(npz['voxel'], axis=0)
            seg= np.expand_dims(npz['seg'], axis=0)#1,100,100,100
        else:
            voxel = np.append(voxel, np.expand_dims(npz['voxel'], axis=0), axis=0)
            seg = np.append(seg, np.expand_dims(npz['seg'], axis=0), axis=0)#x,100,100,100
    print('completed')

    print('resize images')
    dset_test=resize(voxel,seg)
    print('completed')

    dset_test=dset_test.unsqueeze(1)
    test_loader=DataLoader(dset_test,1)
    
    print('eval')
    model=mynet().cuda()
    model.load_state_dict(torch.load('./best_in_mynet.pkl'))
    result=pd.read_csv(testcsv_path)
    modelresult=torch.zeros(117)
    model =model.eval()
    for j, inputs in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        #result['Predicted'][j]=outputs.cpu().detach()
        modelresult[j]=outputs.cpu().detach()
    #result['Predicted'].astype(np.float32)
    result['Predicted']=modelresult.numpy()
    print(result['Predicted'])
    result.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",'--mode', default='test',type=str)
    args = parser.parse_args()
    main(args)




























