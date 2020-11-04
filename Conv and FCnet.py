import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

datadir = "D:\Cats and Dogs images\kagglecatsanddogs_3367a\PetImages"
categories = ["cane","cavallo","elefante","farfalla","gallina","gatto",
              "mucca","pecora","ragno","scoiattolo"]
complete_data = []
img_size = 16

def create_data(datadir,categories):
    for category in categories:
        path = os.path.join(datadir,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(img_arr,(img_size,img_size))
                complete_data.append([new_arr,class_num])
                
            except Exception as e:
                pass
            
create_data(datadir,categories)
tot_data = len(complete_data)

random.shuffle(complete_data)
x = []
y = []

for features,lables in complete_data:
    x.append(features)
    y.append(lables)
    
x = np.reshape(x,(-1,1,img_size,img_size))
y = np.reshape(y,-1)
mask = np.random.choice(tot_data,15000)
x = x[mask]
y = y[mask]
tot_data = 15000

data = {}

def split_data(x,y,num_train=(4*tot_data)//5,val_size=tot_data//5):
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

class ConvFCnet(object):
    
    def __init__(self,input_dims=(1,16,16),hidden_dims=(500,),num_classes=10,reg=0.0,
                 weight_scale=1e-3,dtype=np.float32,filter_size=7,num_filters=32):
            
        self.reg = reg
        self.hidden_dims = hidden_dims
        self.params = {}
        
        C,H,W = input_dims
        F = num_filters
        fs = filter_size
        self.params['W1'] = weight_scale*np.random.randn(F,C,fs,fs)
        self.params['b1'] = np.zeros(F)
        
        p = len(hidden_dims)
        X = F*H*W//4
        for i in range(p):
            if(i==0):
                self.params['W'+str(i+2)] = np.random.randn(X,hidden_dims[0])
                self.params['W'+str(i+2)] *= weight_scale
                
            else:
                self.params['W'+str(i+2)] = np.random.randn(hidden_dims[i-1],hidden_dims[i])
                self.params['W'+str(i+2)] *= weight_scale
                
            self.params['b'+str(i+2)] = np.zeros(hidden_dims[i])
            
        self.params['W'+str(p+2)] = np.random.randn(hidden_dims[p-1],num_classes)
        self.params['W'+str(p+2)] *= weight_scale
        self.params['b'+str(p+2)] = np.zeros(num_classes)
        
        for k,v in self.params.items():
            self.params[k] = v.astype(dtype)
            
    def loss(self,x,y=None):
        
        plen = len(self.hidden_dims)
        
        N,C,H,W = x.shape
        W1 = self.params['W1']
        b1 = self.params['b1']
        F = W1.shape[0]
        fs = W1.shape[2]
        
        conv_param = {'stride':1, 'pad':(fs-1)//2}
        pool_param = {'pool_height':2, 'pool_width':2, 'stride':2}
        
        st1 = conv_param['stride']
        p = conv_param['pad']
        st2 = pool_param['stride']
        ph = pool_param['pool_height']
        pw = pool_param['pool_width']
        
        H1 = 1 + (H + 2*p - fs)//st1
        L1 = 1 + (W + 2*p - fs)//st1
        conv_out = np.zeros((N,F,H1,L1))
        xpad = np.pad(x,((0,0),(0,0),(p,p),(p,p)),'constant',constant_values=0)
        
        for n in range(N):
            for f in range(F):
                for i in range(H1):
                    for j in range(L1):
                        v = xpad[n,:,i*st1:i*st1+fs,j*st1:j*st1+fs]
                        g = v*W1[f,:,:,:]
                        g = g.sum() + b1[f]
                        conv_out[n,f,i,j] = g
                        
        conv_relu_out = np.maximum(conv_out,0)
        maxpool_out = np.zeros((N,F,H//2,W//2))
        w1 = W//2
        
        for n in range(N):
            for f in range(F):
                for i in range(H//2):
                    for k in range(W//2):
                        v = conv_relu_out[n,f,i*st2:i*st2+ph,k*st2:k*st2+pw]
                        g = np.max(v)
                        maxpool_out[n,f,i,k] = g
                        
        affine_in = np.reshape(maxpool_out,(N,-1))
        
        feed_for = affine_in
        cache = {}
        cache['layer'+str(1)] = conv_out
        cache['layer'+str(1)+'wa'] = conv_relu_out
        for i in range(plen):
            W = self.params['W'+str(i+2)] 
            b = self.params['b'+str(i+2)]
            cache['layer'+str(i+2)] = feed_for.dot(W) + b
            feed_for_relu = np.maximum(0,cache['layer'+str(i+2)]) 
            cache['layer'+str(i+2)+'wa'] = feed_for_relu
            feed_for = feed_for_relu
            

        Wf = self.params['W'+str(plen+2)]
        bf = self.params['b'+str(plen+2)]
        scores = feed_for.dot(Wf) + bf
        
        if y is None:
            return scores
        
        loss , grads = 0,{}
        
        scores = np.exp(scores)
        correct_scores = scores[list(range(N)),y]
        scores_sum = np.sum(scores,axis=1)
        loss = -np.sum(np.log(correct_scores/scores_sum))/N
        
        for t in range(plen+2):
            W = self.params['W'+str(t+1)]
            loss += self.reg*(np.sum(W*W))/N
            
        dout = scores/scores_sum.reshape(N,1)
        dout[list(range(N)),y] -= 1
        grads['b'+str(plen+2)] = np.sum(dout,axis=0)/N
        grads['W'+str(plen+2)] = cache['layer'+str(plen+1)+'wa'].T.dot(dout)/N
        grads['W'+str(plen+2)] += self.reg*(self.params['W'+str(plen+2)])
        dtemp = dout.dot(self.params['W'+str(plen+2)].T)
        dtemp[cache['layer'+str(plen+1)+'wa']==0] = 0
        dout = dtemp
        
        for t in range(plen):
            if t==plen-1:
                prev_layer = affine_in
                
            else:
                prev_layer = cache['layer'+str(plen-t)+'wa']
                
            W = self.params['W'+str(plen-t+1)]
            db = np.sum(dout,axis=0)/N
            dW = prev_layer.T.dot(dout)/N
            dW += self.reg*(W)
            dtemp = dout.dot(W.T)
            grads['W'+str(plen-t+1)]= dW
            grads['b'+str(plen-t+1)] = db
            
            if t==plen-1:
                dout = dtemp
                
            else:
                dtemp[prev_layer==0] = 0
                dout = dtemp
                
        daffine = dout
        dmaxpool = daffine.reshape(maxpool_out.shape)
        dconv = np.zeros(conv_out.shape)
        
        for n in range(N):
            for f in range(F):
                for i in range(H//2):
                    for j in range(w1):
                        ind = np.argmax(conv_relu_out[n,f,i*st2:i*st2+ph,j*st2:j*st2+pw])
                        ind1, ind2 = np.unravel_index(ind,(ph,pw))
                        v = dmaxpool[n,f,i,j]
                        dconv[n,f,i*st2:i*st2+ph,j*st2:j*st2+pw][ind1,ind2] = v
                        
        dconv[conv_relu_out==0] = 0
        dW1 = np.zeros(W1.shape) + self.reg*(W1)
        db1 = np.zeros(b1.shape)
        
        for n in range(N):
            for f in range(F):
                db1[f] += dconv[n,f].sum()
                for i in range(H1):
                    for j in range(L1):
                        v = xpad[n,:,i*st1:i*st1+fs,j*st1:j*st1+fs]*dconv[n,f,i,j]
                        dW1[f,:,:,:] += v
                        
            
        dW1 /= N
        db1 /= N
        dW1 += self.reg*(W1)
        grads['W1'] = dW1
        grads['b1'] = db1
        
        return loss,grads
    
class Solver(object):
    
    def __init__(self,model,data,num_epochs,lr_rate,lr_decay,batch_size,
                 num_train_samples,num_val_samples):
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.optim_config = {'t':1}
        self.lr_rate = lr_rate
        
        self.model = model
        
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_val = data['x_val']
        self.y_val = data['y_val']
        
        self.reset()
        
    def reset(self):
        
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_hist = []
        self.train_acc = []
        self.val_acc = []
        
    def adam(self,w,dw,config=None,lr_rate=None,b1=0.9,b2=0.999,eps=1e-8):
        if config is None: config = {}
        config.setdefault('t', 1)
        
        first_mom = 0
        second_mom = 0
        first_mom = b1*first_mom + (1-b1)*dw
        second_mom = b2*second_mom + (1-b2)*dw*dw
        t = config['t'] +1
        
        first_bias = first_mom/(1-b1**t)
        second_bias = second_mom/(1-b2**t)
        next_w = w - lr_rate*first_bias/(np.sqrt(second_bias)+eps)
        config['t'] = t
        
        return next_w,config
    
    def step(self):
        
        num_train = self.x_train.shape[0]
        mask = np.random.choice(num_train,self.batch_size)
        x_batch = self.x_train[mask]
        y_batch = self.y_train[mask]
        
        loss,grads = self.model.loss(x_batch,y_batch)
        self.loss_hist.append(loss)
        
        for k,v in self.model.params.items():
            dw = grads[k]
            config = self.optim_config
            next_w,next_config = self.adam(v,dw,config,self.lr_rate)
            self.model.params[k] = next_w
            self.optim_config = next_config
    def check_acc(self,x,y,num_samples=None,batch_size=32):
        
        N = x.shape[0]
        
        if num_samples is not None and N>num_samples:
            mask = np.random.choice(N,num_samples)
            x = x[mask]
            y = y[mask]
            N = num_samples
            
        num_batches = N//batch_size
        if num_batches*batch_size != N:
            num_batches += 1
            
        y_pred = []
        for t in range(num_batches):
            start = t*batch_size
            end = (t+1)*batch_size
            scores = self.model.loss(x[start:end])
            y_pred.append(np.argmax(scores,axis=1))
            
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred==y)
        
        return acc
    
    def train(self):
        
        num_train = self.x_train.shape[0]
        itr_per_epoch = max(num_train//self.batch_size,1)
        num_itr = itr_per_epoch*self.num_epochs
        
        for i in range(num_itr):
            self.step()
            print('(Iteration %d / %d) loss: %f' % (
                       i + 1, num_itr, self.loss_hist[-1]))
            
            if (i+1)%itr_per_epoch==0:
                self.epoch += 1
                self.lr_rate *= self.lr_decay
                
            if i==0 or (i+1)%itr_per_epoch==0:
                train_acc = self.check_acc(self.x_train,self.y_train,
                                            num_samples=self.num_train_samples)
                val_acc = self.check_acc(self.x_val,self.y_val,
                                         num_samples= self.num_val_samples)
                
                self.train_acc.append(train_acc)
                self.val_acc.append(val_acc)
                print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))
                
                if val_acc>self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    
                    for k,v in self.model.params.items():
                        self.best_params[k] = v.copy()
                        
        self.model.params = self.best_params
        

#train
model = ConvFCnet()
solver = Solver(model,data,20,1e-3,0.9,32,500,500)
solver.train()

#sanity loss check
model = ConvFCnet()
N = 50
x = np.random.randn(N,1,32,32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(x,y)
print('loo with no reg: ', loss)

#overfit small data
num_train = 100
small_data = {
  'x_train': data['x_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'x_val': data['x_val'],
  'y_val': data['y_val'],
}

model = ConvFCnet(weight_scale=1e-2)
solver = Solver(model,small_data,20,1e-3,0.9,32,100,100)
solver.train()

#plot of loss
plt.subplot(2, 1, 1)
plt.plot(solver.loss_hist, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')


#plot of training accuracy andn validation accuracy
plt.subplot(2, 1, 2)
plt.plot(solver.train_acc, '-o')
plt.plot(solver.val_acc, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

        
            
        
        
        
        
                