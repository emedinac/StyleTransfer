# External Hyperparameters
gpus='0' # numbers ('0,1')or '-'
workers = 16
weights = ''
pretrained = False
out_path = 'models/'
dataset = "STL10"

# Network Hyperparameters
architecture = 'Xception'
name_file = 'Aug_Xception'
n_channels = 3
epochs = 200
train_batch = 32
test_batch = 1024

lr = 2e-3 # 1e-6 is the best
momentum = 0.9
weight_decay = 1e-5
init = 'xavier,gauss' # 'uniform,-0.1,0.1'   or   'he,uniform'
# scheduler = '5,10,20,30,50,75-0.2' # schedule - lr decay
scheduler = str(list(range(2,epochs,5)))[1:-1].replace(' ','')+'-0.94' # schedule - lr decay
# scheduler = 'min/0.06/2'
earlystop = False
# Training Structure
type_optimizer = 'Adam' # SGD adam
betas=(0.5, 0.999)
loss = 'CE' # available BCE, DICE,

augmentation = True
# resize = (256,256)
rot = 15 # (-15,-10,-5,0,5,10,15)
trans = ([0.1,0.1],20) # translation and shear
scale = (0.9,1.1)
hflip = 0.5
# vflip = 0.2
color = (0.15, 0.15, 0.15, 0.15)
erase = (0.5, 0.4, 0.3)