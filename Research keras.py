
# coding: utf-8

# In[1]:


import os
import numpy as np
import shutil
import tensorflow as tf

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import scipy
import numpy as np
from pathlib import Path
import os

import keras as k
from keras.layers.core import Layer, Dense
import PIL

from scipy.io.wavfile import read
import pandas as pd
import wave
import pylab
from PIL import Image

from scipy.io import wavfile
#get_ipython().run_line_magic('matplotlib', 'inline')

import librosa
from sklearn.datasets import load_files
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
import time

sample_length = 44100*2 # 2 seconds


# ### Creating dataframes for look-ups

# In[2]:


def labels(path, target_dir):
    df_files = pd.DataFrame(columns=['name', 'target'])
    count = 0
    #print('path: %s  Target:%s'%(path,target_dir))
    for audio in os.walk(path):
        if audio[0][-6:] == "female":
            for aud in audio[2]:
                df_files.loc[count] = [path+"/female/"+aud, 1]
                count += 1
        elif audio[0][-4:] == "male":
            for aud in audio[2]:
                df_files.loc[count] = [path+"/male/"+aud, 0]
                count += 1
    return df_files      


# In[3]:


df_test = labels('/home/vmuser/Documents/Recent/test/', 'test')
df_test = df_test.sample(frac=1).reset_index(drop=True)
print(df_test)

df_valid = labels('/home/vmuser/Documents/Recent/valid/', 'valid') 
df_valid = df_valid.sample(frac=1).reset_index(drop=True)
print(df_valid)

df_train = labels('/home/vmuser/Documents/Recent/train/', 'train')
df_train = df_train.sample(frac=1).reset_index(drop=True)
print(df_train)
#/home/vmuser/Documents/Recent/train/male

# # Converting mp3 files to wav
# 
# I used a bash script to convert mp3 files to wav files which is also attached here, convert.sh.

# ### Following code splits the data into train, test and validation sets. It also extracts the labels from the folder names.
# 
# The data is in 2 folders, female and male. The female folder has the female audios and the male folder has the male audios. The following functions go through the test, train and validation datasets and put the paths and labels in a dataframe based on the folder the data is in.

# ## Functions to split the audio, normalize and draw spectrograms.

# In[4]:


def trim(arr):
    if arr.shape[0] >= sample_length:
        return arr[:sample_length]
    c = np.zeros((sample_length))
    c[:arr.shape[0]] = arr
    return c

def get_spec(arr, framerate):
    spec = pylab.specgram(arr, Fs = framerate)
    X_scaled = preprocessing.scale(spec[0])
    return X_scaled

def get_audio(df):
    data = []
    for i in range(len(df)):
        opened = wave.open(df.at[i, 'name'])
        nchannels, sampwidth, framerate, nframes, comptype, compname =  opened.getparams()
        dataframes = opened.readframes(-1)
        dataframes = np.fromstring(dataframes, 'int32')

        c = trim(dataframes) 
        data.append(c)
    return np.array(data)

def get_spectrograms(df):
    data = []
    for i in range(len(df)):
        opened = wave.open(df.at[i, 'name'])
        nchannels, sampwidth, framerate, nframes, comptype, compname =  opened.getparams()
        dataframes = opened.readframes(-1)
        dataframes = np.fromstring(dataframes, 'int32')

        c = trim(dataframes)
        a = get_spec(c, framerate)
        data.append(a)
    return np.array(data)

def normalize_data(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-299)
    return data


# ### Getting the data and labels

# In[5]:


train_data = normalize_data(get_audio(df_train))
valid_data = normalize_data(get_audio(df_valid))
test_data = normalize_data(get_audio(df_test))

# train_data = train_data.reshape((train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
# valid_data = valid_data.reshape((valid_data.shape[0], valid_data.shape[1]*valid_data.shape[2]))
# test_data = test_data.reshape((test_data.shape[0], test_data.shape[1]*test_data.shape[2]))

train_labels = df_train['target'].values.astype('int')
valid_labels = df_valid['target'].values.astype('int')
test_labels = df_test['target'].values.astype('int')

train_data.shape


# ### GRADIENTS ARE EXPLODING, SEEING THIS THROUGH NAN VALUES AND EXTREMELY LARGE VALUES, try using RELU function 

# In[6]:


class RBM:
    def __init__(self, visible_number, hidden_number=6, theta=None, hidden_bias=None, visible_bias=None, hidden=None):
        self.visible_number = visible_number
        self.hidden_number = hidden_number
        self.free_energy = []
    
        self.theta = np.random.random_sample((visible_number, hidden_number)) if theta is None else theta
        self.theta = self.array_to_float32_tensor(self.theta)
        self.hidden_bias = np.ones((1,hidden_number)) if hidden_bias is None else hidden_bias
        self.hidden_bias = self.array_to_float32_tensor(self.hidden_bias)
        self.visible_bias = np.ones((1,visible_number)) if visible_bias is None else visible_bias
        self.visible_bias = self.array_to_float32_tensor(self.visible_bias)
        self.hidden = np.ones((visible_number,hidden_number)) 
        self.hidden = self.array_to_float32_tensor(self.hidden)
        
    def array_to_float32_tensor(self,arr,msg=None):
        x=tf.cast(tf.convert_to_tensor(arr), tf.float64)
        return x
    
    def probabilities(self, values):
        val1 = tf.random_uniform(values.get_shape())
        val1 = self.array_to_float32_tensor(val1)
        val = k.backend.sigmoid(tf.sign(values - val1))
        return val
    
    def forward_batch(self, visible):
        a = self.array_to_float32_tensor(visible)
        m = k.backend.dot(a, self.theta)
        h = tf.add(m, self.hidden_bias)
#         h = k.backend.relu(h, alpha=0.0, max_value=None)
        return k.backend.sigmoid(h)
    
    def forward_propagation(self, visible):
        a=self.array_to_float32_tensor(visible)
        b=self.array_to_float32_tensor(self.theta)
        hb=self.array_to_float32_tensor(self.hidden_bias)
        hidden = tf.add(k.backend.dot(a,b),hb)
        activate_hidden = k.backend.sigmoid(hidden)
#         activate_hidden = tf.nn.convolution(activate_hidden, w, strides=[1, 1, 2, 2], padding='SAME',data_format='NCHW')
#         activate_hidden = k.backend.sigmoid(hidden)
        return hidden, activate_hidden
    
    def backward_propagation(self, hidden): 
        a = self.array_to_float32_tensor(hidden)
        b = self.array_to_float32_tensor(k.backend.transpose(self.theta))
        dot=k.backend.dot(a, b)
        visible = tf.add(dot,self.array_to_float32_tensor(self.visible_bias))
        activate_visible=k.backend.sigmoid(visible)
#         activate_visible=k.backend.sigmoid(visible)
        
        return visible, activate_visible

    def gibbs_sampling_h_given_v(self, sample_v):
        l = 0.9
        _, forward_prop = self.forward_propagation(sample_v)
        sample_h = self.probabilities(forward_prop)
        return sample_h,forward_prop
    
    
    def gibbs_sampling_v_given_h(self, sample_h):
        l = 0.9
        _, backward_prop = self.backward_propagation(sample_h)
        sample_v = self.probabilities(backward_prop)
        return sample_v, backward_prop
    
    def gibbs_samplingh(self, sample_h):
        sample_v, _ = self.gibbs_sampling_v_given_h(sample_h)
        return self.gibbs_sampling_h_given_v(sample_v)

    def gibbs_samplingv(self, sample_v):
        sample_h, _ = self.gibbs_sampling_h_given_v(sample_v)
        return self.gibbs_sampling_v_given_h(sample_h)  
    
    def visible_sampling(self, visible, n):
        hidden_sample, _ = self.gibbs_sampling_h_given_v(visible)
        visible_sample, _ = self.gibbs_sampling_v_given_h(hidden_sample)
        for i in range(n-1):
            hidden_sample, _ = self.gibbs_sampling_h_given_v(visible_sample)
            visible_sample, _ = self.gibbs_sampling_v_given_h(hidden_sample)
        return visible_sample, hidden_sample
                                                
    def Contrastive_Divergence(self, visible, alpha, n):
        visible_sample, hidden_sample = self.visible_sampling(visible, n)
        _, hidden = self.forward_propagation(visible)
        self.hidden = hidden
        self.hidden_bias += alpha * k.backend.mean(hidden - hidden_sample)
        self.visible_bias += alpha * k.backend.mean(visible- visible_sample)
        n = 1
        h = tf.cast(k.backend.transpose(hidden), tf.float64)
        visible = tf.cast(visible, tf.float64)
        hidden_sample = tf.convert_to_tensor(hidden_sample)
        visible_sample = tf.convert_to_tensor(visible_sample)
        _ = visible.get_shape().as_list()[0]
        a = tf.divide(tf.matmul(h, visible), _)
        b  = tf.matmul(k.backend.transpose(hidden_sample), visible_sample)
        c = tf.subtract(a,b)
        self.theta += k.backend.transpose(alpha * c)
        return  self.theta, self.visible_bias, self.hidden_bias, self.hidden
                                       
        
    def energy_function(self, visible):
        visible_sample, _ = self.visible_sampling(visible,1)
        visible_term = k.backend.dot(visible_sample, k.backend.transpose(self.visible_bias))
        weights_term = tf.add(k.backend.dot(visible_sample, self.theta), self.hidden_bias)
        sum_term = k.backend.sum(k.backend.log(1 + k.backend.exp(weights_term)))
        free_energy = -visible_term - sum_term
        self.free_energy.append(k.backend.sum(free_energy))  
        return free_energy
    
    def reconstruction(self, visible):
        hidden, activate_hidden = self.forward_propagation(visible)
        return self.backward_propagation(hidden)
    
    def reconstruction_error(self, data):
        h = gibbs_samplingv(data)
        v = gibbs_samplingh(h)
        difference = tf.stop_gradient(data - v)
        error = tf.reduce_sum(err * err)
        return error

    def training(self, visible, epochs=3, alpha=0.01):
        epochs = epochs if epochs > 0 else 1
        for i in range(epochs):
            self.Contrastive_Divergence(visible, alpha, 1)
        return
    
    # TODO try different costs
    #             save_path = saver.save(sess, 'model/my_test_model', global_step=10, write_meta_graph=True)
    #             self.cost = tf.sqrt(tf.reduce_mean(
    #             tf.square(tf.subtract(self.inputs, self.reconstruction))))
    #             tf.scalar_summary("train_loss", self.cost)
    #             self.summary = tf.merge_all_summaries()
    
    def show_energy(self):
        plt.figure(figsize=(12,15))
        plt.plot(self.free_energy)
        plt.title('Free Energy')
        plt.xlabel('epochs')
        plt.ylabel('Energy')
        plt.plot()

    def Something_about_Cost(self, visibles, n=1):
        for i in range(n):
            if i == 0:
                visible_sample, pre_v = self.gibbs_samplingv(visibles)
            else:
                visible_sample, pre_v = self.gibbs_samplingv(visible_sample)
        visible_sample = k.backend.stop_gradient(visible_sample)
        recon_loss = -k.backend.mean(k.backend.sum(visibles * k.backend.log(k.backend.sigmoid(pre_v)) + k.backend.log(1 - k.backend.sigmoid(pre_v))))
        contra_div_loss = k.backend.mean(self.energy_function(visibles)) - k.backend.mean(self.energy_function(visible_sample))
        return recon_loss, contra_div_loss
    
    def check_overfitting(self, visible_train, visible_test):
        ratio = k.backend.exp(-self.energy_function(visible_train) + self.energy_function(visible_test))
        if r > 1:
            print("overfitting")
            
        else:
            print("not overfitting") 
                                                
    def Pseudo_Likelihood(self, visibles):
        visible_data_energy_function = self.energy_function(visibles)
        print(visible_data_energy_function)
        corrupted = visibles.copy()
        index = (np.arange(visibles.shape[0]),  np.random.randint(0, visibles.shape[1], visibles.shape[0]))
        corrupted[index] = 1 - corrupted[index]
        visible_data_energy_function = self.energy_function(visibles)
        print(visible_data_energy_function)
        corrupted_energy_function = self.energy_function(corrupted)
        print(corrupted_energy_function)
        cost = k.backend.log(k.backend.sigmoid(corrupted_energy_function - visible_data_energy_function ))
        print('cost', cost)
        return cost

    def show_energy(self):
        plt.figure(figsize=(12,15))
        plt.plot(self.free_energy)
        plt.title('Free Energy')
        plt.xlabel('epochs')
        plt.ylabel('Energy')
        plt.plot()


# In[7]:


class DBN: # A DNB will consist of 2 RBMs
    def __init__(self, visible):
        # learning_rate = 0.01, n = 1
        self.visible = visible

    def train(self):
        a = time.time()
        rbm1 = RBM(visible_number=88200, hidden_number=50)
        rbm2 = RBM(visible_number=50, hidden_number=2)
#         rbm3 = RBM(visible_number=20000, hidden_number=100)
#         rbm4 = RBM(visible_number=10000, hidden_number=50)
        
        num_epochs = 5
        h = None
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
#             batch = mnist.train.next_batch(50)
            sess.run(init)
            b = time.time()
            print("%f minutes to init tensorflow"%((b-a)/60.0))
            a1 = time.time()
            for epoch in range(num_epochs):
                sess.run(rbm1.Contrastive_Divergence(self.visible, 0.01, 1))
                np.save("theta_1.npy", rbm1.theta.eval(session=sess))
                np.save("hidden_bias_1.npy", rbm1.hidden_bias.eval(session=sess))
                np.save("visible_bias_1.npy", rbm1.visible_bias.eval(session=sess))
            
            h = rbm1.forward_batch(self.visible)
            h = h.eval(session=sess)
            b1 = time.time()
            print("%f minutes to train RBM 1"%((b1-a1)/60.0))
            
            a2 = time.time()
            for epoch in range(num_epochs):
                sess.run(rbm2.Contrastive_Divergence(h, 0.01, 1))
                np.save("theta_2.npy", rbm2.theta.eval(session=sess))
                np.save("hidden_bias_2.npy", rbm2.hidden_bias.eval(session=sess))
                np.save("visible_bias_2.npy", rbm2.visible_bias.eval(session=sess))
                np.save("hidden.npy", rbm2.hidden.eval(session=sess))
            b2 = time.time()
            print("%f minutes to train RBM 2"%((b2-a2)/60.0))
            
#             a3 = time.time()
#             for epoch in range(num_epochs):
#                 sess.run(rbm3.Contrastive_Divergence(h, 0.01, 1))
#                 np.save("theta_3.npy", rbm3.theta.eval(session=sess))
#                 np.save("hidden_bias_3.npy", rbm3.hidden_bias.eval(session=sess))
#                 np.save("visible_bias_3.npy", rbm3.visible_bias.eval(session=sess))
#             b3 = time.time()
#             print("%f minutes to train RBM 3"%((b3-a3)/60.0))
            
#             a4 = time.time()
#             for epoch in range(num_epochs):
#                 sess.run(rbm4.Contrastive_Divergence(h, 0.01, 1))
#                 np.save("theta_4.npy", rbm4.theta.eval(session=sess))
#                 np.save("hidden_bias_4.npy", rbm4.hidden_bias.eval(session=sess))
#                 np.save("visible_bias_4.npy", rbm4.visible_bias.eval(session=sess))
#             b4 = time.time()
#             print("%f minutes to train RBM 4"%((b4-a4)/60.0))
            
    def transform(self, data):
        a = time.time()
        t1 = np.load("theta_1.npy")
        print(t1)
        h1 = np.load("hidden_bias_1.npy")
        v1 = np.load("visible_bias_1.npy")
        
        t2 = np.load("theta_2.npy")
        h2 = np.load("hidden_bias_2.npy")
        v2 = np.load("visible_bias_2.npy")
        
        hidden = np.load("hidden.npy")
        
#         t3 = np.load("theta_3.npy")
#         h3 = np.load("hidden_bias_3.npy")
#         v3 = np.load("visible_bias_3.npy")
        
#         t4 = np.load("theta_4.npy")
#         h4 = np.load("hidden_bias_4.npy")
#         v4 = np.load("visible_bias_4.npy")
        
        
        b = time.time()
        print("%f minutes to load data"%((b-a)/60.0))
        
        rbm1 = RBM(visible_number=88200, hidden_number=50, theta=t1, hidden_bias=h1, visible_bias=v1, hidden=hidden)
        rbm2 = RBM(visible_number=50, hidden_number=2, theta=t2, hidden_bias=h2, visible_bias=v2, hidden=hidden)
#         rbm3 = RBM(visible_number=20000, hidden_number=100, theta=t3, hidden_bias=h3, visible_bias=v3)
#         rbm4 = RBM(visible_number=10000, hidden_number=50,theta=t4, hidden_bias=h4, visible_bias=v4)
        
        
        h = None
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            a1 = time.time()
            h = rbm1.forward_batch(data).eval(session=sess)
            h = rbm2.forward_batch(h).eval(session=sess)
#             h = rbm3.forward_batch(h).eval(session=sess)
#             h = rbm4.forward_batch(h).eval(session=sess)
            b1 = time.time()
            print("%f minutes to do forward propaganda"%((b1-a1)/60.0))
        return h, hidden


# In[ ]:


dbn = DBN(train_data)
dbn.train()


# In[9]:


#train_h = dbn.transform(train_data)
valid_h, hidden = dbn.transform(valid_data)
test_h, hidden = dbn.transform(test_data)

#print(train_h)
#print(valid_h)
# print(test_h)


# In[10]:


t1 = np.load("theta_1.npy")
h1 = np.load("hidden_bias_1.npy")
v1 = np.load("visible_bias_1.npy")


# In[11]:


v1


# In[12]:


t2 = np.load("theta_2.npy")
h2 = np.load("hidden_bias_2.npy")
v2 = np.load("visible_bias_2.npy")


# In[13]:


v2


# ## Implementing some Binary Classifiers

# In[16]:


#Support Vector Machines as classifier
def SVM(data, label, test_data, test_labels):
    clf = svm.SVC(kernel='rbf')
    clf.fit(data, label+0.0)
    predictions = clf.predict(test_data)
    print("Accuracy:", metrics.accuracy_score(test_labels, predictions))
    print("Precision:",metrics.precision_score(test_labels, predictions))
    print("Recall:",metrics.recall_score(test_labels, predictions))

print("validation data ", SVM(hidden, train_labels, valid_h, test_labels))
print("training data ", SVM(hidden, valid_labels, hidden, valid_labels))


# In[17]:


# TODO: logistic regression
from sklearn.linear_model import LogisticRegression
def Logistic_Regression(data, label, test_data, test_labels):
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    clf.fit(data, label+0.0)
    predictions = clf.predict(test_data)
#clf.score(train_h, train_labels)
    print("Accuracy:", metrics.accuracy_score(test_labels, predictions))
    print("Precision:",metrics.precision_score(test_labels, predictions))
    print("Recall:",metrics.recall_score(test_labels, predictions))
print("validation data ",  Logistic_Regression(hidden, valid_labels, test_h, test_labels))
print("training data ",  Logistic_Regression(hidden, valid_labels, hidden, valid_labels))    


# # DO NOT RUN THE FOLLOWING CODE!!!
# 
# The code takes the data that has the right labels which is the gender and puts it in a male and female folder. Reason for this is so that it can be easy to get the labels once I start classifying.

# In[ ]:


# my_data = np.genfromtxt('/files1b/856182/cv_corpus_v1/cv-valid-train.csv',dtype='unicode',delimiter=',')

# good_files = []
# female = []
# male = []
# num_files =  len(my_data)
# for i in range(num_files):
#     if my_data[i][5] == '':
#         continue
#     elif my_data[i][5] == 'female':
#         female.append(my_data[i])
#     elif my_data[i][5] == 'male':
#         male.append(my_data[i]) 

# for i in range(len(male)):
#     shutil.copy('/files1b/856182/cv_corpus_v1/'+male[i][0],'/files1b/856182/cv_corpus_v1/datasets/male/')
#     print ("File number "+str(i)+" out of "+str(len(male)))
    
# for i in range(len(female)):
#     shutil.copy('/files1b/856182/cv_corpus_v1/'+female[i][0],'/files1b/856182/cv_corpus_v1/datasets/female/')
#     print ("File number "+str(i)+" out of "+str(len(female)))

