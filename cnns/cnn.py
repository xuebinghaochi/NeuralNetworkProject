import mnist
import numpy as np
from mnist import train_images

from cnns import maxpool
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

train_images=mnist.train_images(target_dir="/Users/qs/PyCharmMiscProject/cnns/data/mnist/MNIST/raw")[:1000]
train_labels=mnist.train_labels(target_dir="/Users/qs/PyCharmMiscProject/cnns/data/mnist/MNIST/raw")[:1000]
test_images = mnist.test_images(target_dir="/Users/qs/PyCharmMiscProject/cnns/data/mnist/MNIST/raw")[:1000]#手动添加了target_dir参数，默认None，不影响正常调用
test_labels = mnist.test_labels(target_dir="/Users/qs/PyCharmMiscProject/cnns/data/mnist/MNIST/raw")[:1000]

conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13*13*8,10)

def forward(image,label):
    out=conv.forward((image/255)-0.5)
    out=pool.forward(out)
    out=softmax.forward(out)

    loss=-np.log(out[label])
    acc=1 if np.argmax(out)==label else 0
    return out,loss,acc

def train(im,label,lr=.005):
    out,loss,acc=forward(im,label)
    gradient=softmax.backprop(out,label,lr)
    gradient=pool.backprop(gradient)
    gradient=conv.backprop(gradient,lr)
    return loss,acc

print('MNIST CNN initialized!')

for epoch in range(3):
    print("--- Epoch %d ---" % (epoch+1))
    permutation = np.random.permutation(len(train_images))#随机化train_images图片序列
    train_images=train_images[permutation]
    train_labels=train_labels[permutation]

    loss=0
    num_correct=0
    for i,(im,label) in enumerate(zip(train_images,train_labels)):
        if i%100==99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i+1,loss/100,num_correct)
            )
            loss=0
            num_correct=0

        l,acc=train(im,label,lr=0.01)
        loss+=l
        num_correct+=acc

print('\n--- Testing the CNN ---')
loss=0
num_correct=0
for im,label in zip(test_images,test_labels):
    _,loss,acc=forward(im,label)
    loss+=loss
    num_correct+=acc
num_test=len(test_images)
print('Test Loss:',loss/num_test)
print('Test Accuracy:',num_correct/num_test)