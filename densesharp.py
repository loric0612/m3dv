import torch
import torch.nn as nn

class dense_block(nn.Module):
    def __init__(self, channel):
        super(dense_block, self).__init__()
        self.tr1 = nn.Sequential(
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, 64, kernel_size=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 16,kernel_size=(3,3,3), padding=(1,1,1))
        )
        self.tr2 = nn.Sequential(
            nn.BatchNorm3d(channel+16),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel+16, 64, kernel_size=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 16, kernel_size=(3,3,3), padding=(1,1,1))
        )
        self.tr3 = nn.Sequential(
            nn.BatchNorm3d(channel+32),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel+32, 64,kernel_size=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 16, kernel_size=(3,3,3), padding=(1,1,1))
        )
        self.tr4 = nn.Sequential(
            nn.BatchNorm3d(channel+48),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel+48, 64,kernel_size=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 16, kernel_size=(3,3,3), padding=(1,1,1))
        )

    def forward(self, x):
        x1 = self.tr1(x)
        x = torch.cat((x, x1), dim=1)
        x1 = self.tr2(x)
        x = torch.cat((x, x1), dim=1)
        x1 = self.tr3(x)
        x = torch.cat((x, x1), dim=1)
        x1 = self.tr4(x)
        x = torch.cat((x, x1), dim=1)
        return x




class trans_block(nn.Module):
    def __init__(self, shape,final_flag=0):
        super(trans_block, self).__init__()
        if (final_flag==0):
            self.trans = nn.Sequential(
                nn.BatchNorm3d(2*shape),
                nn.ReLU(inplace=True),
                nn.Conv3d(2*shape, shape, kernel_size=(1,1,1), padding=(1,1,1)),
                nn.AvgPool3d(kernel_size=(2,2,2))
            )
        else:
            self.trans = nn.Sequential(
                nn.BatchNorm3d(2*shape),
                nn.ReLU(inplace=True),
                nn.AvgPool3d(kernel_size=(8,8,8))
            )
    def forward(self, x):
        x = self.trans(x)
        return x




class mynet(nn.Module):
    def __init__(self,shape=32):#shape=32 in below
        super(mynet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3,3,3), padding=(1,1,1))
        self.dense1 = dense_block(32)#32,48,64,80
        self.trans1 = trans_block(48)#80+16 /2
        self.dense2 = dense_block(48)#48,64,80,96
        self.trans2 = trans_block(56)#96+16 /2
        self.dense3 = dense_block(56)#56,72,88,104
        self.trans3 = trans_block(60,1)#104+16 /2
        
        self.output = nn.Sequential(
            nn.Linear(120,1),
            #nn.Dropout(p=0.5),
            #nn.ReLU(inplace=True),
            #nn.Linear(120, 60),
            #nn.Dropout(p=0.5),
            #nn.ReLU(inplace=True),
            #nn.Linear(60, 1),
            nn.Sigmoid()#
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = x.view(x.shape[0], -1).squeeze()#sigmoid之前reshape
        x = self.output(x)
        return x.squeeze()