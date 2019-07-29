import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
import torch, torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn

from TrainingEnvironment import *
from Networks import classification as nets
from termcolor import colored
class fully_train:
    def __init__(self, net, conf):
        self.conf = conf
        self.net = net
        self.best_acc = 0
        ################# GPU CONFIGURATION #################
        if self.conf.gpus:
            self.net.cuda()
            cudnn.benchmark = True
            gpu_list = [int(i) for i in self.conf.gpus.split(',')]
            if len(gpu_list)>1:
                self.DataParallelEnable = True
                self.net = nn.DataParallel(net, device_ids=gpu_list)
                print("Let's use ", len(gpu_list), " GPUs!")
            else:
                self.DataParallelEnable = False
                print("Using only the GPU : ", gpu_list[0])
        else:
            print('GPUs are not being used.')
        if not os.path.isdir(self.conf.out_path):
            os.mkdir(self.conf.out_path)
            os.chmod(self.conf.out_path, 0o777)
        ################# CONFIGURATION #################
        Load_loaders_and_optimization(self) # Env creation: optimizer, Dataloader (data augmentation), ...
        self.N_train = self.trainloader.__len__()
        self.N_test = self.testloader.__len__()
        print('Number of batches for training: ', self.N_train)
        print('Number of batches for testing: ', self.N_test)
        ################# SAVE FILES #################
        self.name_to_save = self.conf.name_file
        self.name_file_temp = self.conf.name_file+"_"+str(0).zfill(3)
        num=0
        if os.path.exists(self.conf.out_path + self.name_file_temp+'.pth'):
            while(os.path.exists(self.conf.out_path + self.name_file_temp+'.pth')):
                self.name_file_temp = self.conf.name_file+"_"+str(num).zfill(3)
                num += 1
        self.name_file = self.name_file_temp
        self.num = str(num).zfill(3)
        open(self.conf.out_path + self.name_file + '.pth', 'w').close()
        print(colored("future file :   " + self.conf.out_path + self.name_file + '.pth','yellow'))
        
    def save_model(self):
        torch.save({'arch': self.name_file,
                    'state_dict': self.net.module.state_dict() if self.DataParallelEnable else self.net.state_dict(),
                    'best_prec': self.best_acc,
                    'epoch': self.best_epoch,
                    'optimizer' : self.optimizer.state_dict(),
                    }
                    , self.conf.out_path + self.name_file+'.pth')
        print(colored('Checkpoint {:04} saved !'.format(self.best_epoch), 'green'))
    
    def load_model(self, model):
        checkpoint = torch.load(model)
        if self.DataParallelEnable: self.net.module.load_state_dict(checkpoint['state_dict'])
        else: self.net.load_state_dict(checkpoint['state_dict'])

    def train_model(self):
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        for epoch in range(self.conf.epochs):
            print('Starting epoch {}/{}'.format(epoch + 1, self.conf.epochs))
            train_loss, train_acc = training_process(self, epoch)
            # print('   ===> Training ===> Loss: {0:.6f} - Acc: {1:.6f}'.format( train_loss/self.N_train, train_acc/50)) # For STL-10 100*Val/5000
            print('   ===> Training ===> Loss: {0:.6f} - Acc: {1:.6f}'.format( train_loss/self.N_train, train_acc/self.N_train))
            test_loss, test_acc = evaluation_process(self, epoch)
            # print('   ===> Evaluation ===> Loss: {0:.6f} - Acc: {1:.6f}'.format( test_loss/self.N_test, test_acc/80)) # For STL-10 100*Val/8000
            print('   ===> Evaluation ===> Loss: {0:.6f} - Acc: {1:.6f}'.format( test_loss/self.N_test, test_acc/self.N_test))

            test_acc = test_acc/self.N_test
            train_acc = train_acc/self.N_train
            self.train_losses.append(train_loss/self.N_train)
            self.test_losses.append(test_loss/self.N_test)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            if self.best_acc<test_acc:
                self.best_epoch = int(epoch + 1)
                self.best_acc = test_acc
                self.train_acc = train_acc
                self.save_model()

    def CloseLogger(self, text='', interrump=""):
        dicts={'best train acc':self.train_acc,
                'best test acc':self.best_acc,
                'epoch':self.best_epoch, 
                'experiment': self.conf.dataset+"_"+self.num}
        file={'train acc ' + self.name_file  +"_"+self.num:     self.train_accs,
                'test acc ' + self.name_file +"_"+self.num:         self.test_accs,
                'train loss '+ self.name_file+"_"+self.num:     self.train_losses,
                'test loss '+ self.name_file +"_"+self.num:         self.test_losses,
                }
        rlog.save_log(dicts=dicts, file=file, ShortName_in=interrump+self.name_to_save, ExperimentName_in=self.name_file+text, GeneralComments_in='')


# Early stopping
if __name__ == '__main__':
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    import config_classification as conf
    os.environ["CUDA_VISIBLE_DEVICES"]=conf.gpus # USED ONLY IF OTHER GPUS ARE BEING USED
    import utils.rlog_creator as rlog
    from importlib import reload;reload(rlog) # not import for python2
    net = nets.ChooseNet(conf.architecture, pretrained=conf)  # Choose the netural network in this function Nerworks/Xception.py -- 
    learning = fully_train(net, conf)
    try:
        learning.train_model()
        print(colored("This is the best result: ", 'green'))
        print("     ", learning.best_epoch,": ", learning.best_acc)
        learning.CloseLogger(text = '   -  Simple test for Xception using the STL-10')
    except:
        print(colored("This is the best result: ", 'green'))
        print("     ", learning.best_epoch,": ", learning.best_acc)
        learning.CloseLogger(text = '   -  Simple test for Xception using the STL-10  (INTERRUPTED!!!!!)')