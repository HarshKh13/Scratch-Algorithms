import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

datadir = "D:\Cats and Dogs images\kagglecatsanddogs_3367a\PetImages"
categories = ["cane","cavallo","elefante","farfalla","gallina",
              "gatto","mucca","pecora","ragno","scoiattolo"]
complete_data = []
img_size = 32

def create_data(datadir,categories):
    for category in categories:
        path = os.path.join(datadir,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_img = cv2.resize(img_arr,(img_size,img_size))
                complete_data.append([new_img,class_num])
                
            except Exception as e:
                pass
            
create_data(datadir,categories)
tot_data = 25000
mask = np.random.choice(len(complete_data),tot_size)
complete_data = complete_data[mask]

random.shuffle(complete_data)
x = [], y = []

for feature,lables in complete_data:
    x.append(features)
    y.append(lables)
    
x = np.reshape(x,(-1,1,img_size,img_size))
y = np.reshape(y,-1)

data = []

def split_data(x,y,train_size=(4*tot_data)/5,val_size=tot_data/5):
    mask = list(range(num_train))
    x_train = x[mask]
    y_train = y[mask]
    
    mask = list(range(num_train,tot_data))
    x_val = x[mask]
    y_val = y[mask]
    
    return x_train,y_train,x_val,y_val

x_train,y_train,x_val,y_val = split_data(x,y)
data['x_train'], data['y_train'] = x_train, y_train
data['x_val'], data['y_val'] = x_val, y_val

class ThreelayerConvnet(object):
    
    def __init__(input_dims=1,32,32,num_filters=32,filter_size=7,weight_scale=1e-3
                 hidden_dims=100,num_classes=10,reg=0.0,dtype=np.float32):
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        C,H,W = input_dims
        F = num_filters
        fs = filter_size
        self.params['W1'] = np.random.randn(F,C,fs,fs)*weight_scale
        self.params['b1'] = np.random.randn(F)
        self.params['W2'] = np.random.randn(F*H*W//4,hiddden_dims)*weight_scale
        self.params['b2'] = np.random.randn(hidden_dims)
        self.params['W3'] = np.random.randn(hidden_dims,num_classes)
        self.params['W3'] *= weight_scale
        self.params['b3'] = np.random.randn(num_classes)
        
        for k,v in self.params.items():
            self.params[k] = v.astype(dtype)
            
    def loss(self,x,y):
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        conv_param = {'stride':2, 'pad':2}
        pool_param = {'pool_height':2, 'pool_width':2, 'stride':2}
        
        st1 = conv_param['stride']
        p = conv_param['pad']
        st2 = pool_param['stride']
        ph = pool_param['pool_height']
        pw = pool_param['pool_width']
        
        N,C,H,W = x.shape
        F = W1.shape[0]
        fs = W1.shape[2]
        H1 = (H + 2*p - fs)//st1
        W1 = (W + 2*p - fs)//st1
        conv_out = np.zeros((N,F,H1,W1))
        xpad = np.pad(x,((0,0),(0,0),(p,p),(p,p)),'constant',constant_values=0)
        
        for n in range(N):
            for f in range(F):
                for i in range(H1):
                    for j in range(W1):
                        v = xpad[n,:,i*st1:i*st1+fs,j*st1:j*st1+fs]
                        g = v*W1[f,:,:,:]
                        g = g.sum() + b1[f]
                        conv_out[n,f,i,j] = g
                        
        conv_relu_out = np.maximum(conv_out,0)
        maxpool_out = np.zeros((N,F,H/2,W/2))
        
        for n in range(N):
            for f in range(F):
                for i in range(H/2):
                    for j in range(W/2):
                        v = conv_relu_out[n,f,i*st2;i*st2+ph,j*st2:j*st2+pw]
                        g = np.max(v)
                        max_pool_out[n,f,i,j] = g
                        
        affine_in = np.reshape(max_pool_out,(N,-1))
        first_layer = affine_in.dot(W2) + b2
        first_layer_relu = np.maximum(first_layer,0)
        scores = first_layer_relu.dot(W3) + b3
        
        if y is None:
            return scores
        
        loss = 0, grads = {}
        
        scores = np.exp(scores)
        correct_scores = scores[list(range(N)),y]
        scores_sum = np.sum(scores,axis=0)
        loss = -np.sum(np.log(correct_scores/scores_sum))/N
        loss += self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(w3*W3))/N
        dout = scores/scores_sum.reshape(N,1)
        db3 = np.sum(dout,axis=1)/N
        dW3 = first_layer_relu.dot(dout.T) + self.reg*(W3)
        dfirst_layer = dout.dot(W3.T)
        dfirst_layer[first_layer_relu==0] = 0
        db2 = np.sum(dfirst_layer,axis=1)/N
        dW2 = affine_in.dot(dfirst_layer.T) + self.reg*(W2)
        daffine = dfirst_layer.dot(W2.T)
        
        dmaxpool = np.reshape(affine_in,maxpool_out.shape)
        dconv = np.zeros(conv_out.shape)
        
        for n in range(N):
            for f in range(F):
                for i in range(H/2):
                    for j in range(W/2):
                        ind = np.argmax(conv_relu_out[n,f,i*st2;i*st2+ph,j*st2:j*st2+pw])
                        ind1, ind2 = np.unravel_index(ind,(ph,pw))
                        v = dmaxpool[n,f,i,j]
                        dconv[n,f,i*st2:i*st2+ph,j*st2:j*st2+pw][ind1,ind2] = v
                        
        dconv[conv_relu_out==0] = 0
        dW1 = np.zeros(W1.shape) + self.reg*(W1)
        db1 = np.zeros(b1.shape)
        dxpad = np.zeros(xpad.shape)
        
        for n in range(N):
            for f in range(F):
                db1[f] += dconv[n,f].sum()
                for i in range(H1):
                    for j in range(W1):
                        v = xpad[n,f,i*st1:i*st1+fs,j*st1:j*st1+fs]*dconv[n,f,i,j]
                        g = W1[f,:,:,:]*dconv[n,f,i,j]
                        dW1[f,:,:,:] += v
                        dxpad[n,f,i*st1:i*st1+fs,j*st1:j*st1+fs] += g
                        
        dx = dxpad[:,:,p:p+H,p:p+W]
        
        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2
        grads['W3'], grads['b3'] = dW3, db3
        
        return loss,grads
    
class Solver(object):
    
    def __init__(self,model,data,num_epochs,lr_decay,
                 batch_size,num_train_samples,num_val_samples):
        
        self.model = model
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_val = data['x_val']
        self.y_val = data['y_val']
        
        self.num_epochs = num_epochs
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = nunm_val_samples
        
    def reset(self):
        
        self.epoch = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
    def sgd(self,w,dw,learning_rate=):
        self.learning_rate = learning_rate
        w -= learning_rate*dw
        return w
        
    def step(self):
        
        num_train = self.x_train.shape[0]
        mask = np.random.choice(num_train,batch_size)
        x_batch = x_train[mask]
        y_batch = y_trainn[mask]
        
        loss,grads = self.model.loss(x_batch,y_batch)
        self.loss_history.append(loss)
        
        for p,w in self.model.params.items():
            dw = grads[w]
            next_w = sgd(w,dw)
            self.params[p] = next_w
            
    def check_accuracy(self,x,y,num_samples=None,batch_size=):
        
        N = x.shape[0]
        if num_samples is not None and N>num_samples:
            mask = np.random.choice(N,num_samples)
            N = num_samples
            x = x[mask]
            y = y[mask]
            
        num_batches = N//batch_size
        if(num_batches*batch_size<N):
            num_batches += 1
            
        y_pred = []
        for i in range(num_batches):
            start = i*batch_size
            end = (i+1)*batch_size
            scores = self.model.loss(x[start:end])
            y_pred.append(np.argmax(scores,axis=1))
            
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred==y)
        
        return acc
    
    def train(self):
        
        num_train = self.x_train.shape[0]
        iterations_per_epoch = max(num_train//self.batch_size, 1)
        num_iterations = self.num_epochs*iterations_per_epoch
        
        for i in range(num_iterations):
            self.step()
            print('(Iteration %d / %d) loss: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))
            
            epoch_end = (i+1)%iterations_per_epoch
            if epoch_end==0:
                self.epoch += 1
                self.learning_rate *= self.lr_decay
                
            if t==0 or epoch_end==0:
                train_acc = check_acccuracy(x_train,y_train,
                                            num_samples=self.num_train_samples)
                val_acc = check_accuracy(x_val,y_val,
                                         num_samples= self.num_val_samples)
                
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))
                
                if val_acc>self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    
                    for k,v in self.model.params.items():
                        self.best_params[k] = v.copy()
                        
        self.model_params = self.best_params
        
        
            

        
        
        
                         
        
        
            

