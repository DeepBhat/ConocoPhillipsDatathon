#!/usr/bin/env python
# coding: utf-8

# ### Importing the dataset

# In[119]:


import pandas as pd
# read the file
train_set = pd.read_csv("equip_failures_training_set.csv")
y = train_set.iloc[:, 1].values


# ### Adding na columns

# In[ ]:


# adding the binary na columns for each sensor value
for key in train_set:
    if "sensor" in key:
        train_set[key + "_na"] = [1 if train_set[key][i] == "na" else 0 for i in range(len(train_set[key]))]


# 

# In[ ]:


X = train_set.iloc[:, 2:].values
import numpy as np
X = np.where(X == "na", 0, X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ### Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)


# ### Building the ANN model

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 170, kernel_initializer = 'uniform', activation = 'relu', input_dim = 340))

## Adding the convolution layer
#classifier.add(Conv1D(10, (3), activation = 'tanh'))

# Adding the next five hidden layers
for i in range(3):
    classifier.add(Dense(units = 170, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.2))

    
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

adamOp = keras.optimizers.Adam(lr = 0.0005, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, amsgrad = False)

# Compiling the ANN
classifier.compile(optimizer = adamOp, loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Fitting the ANN normally

# In[ ]:


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50, validation_split = 0.1)


# ### Fitting the ANN with GridSearchCV

# In[ ]:





# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[ ]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


# ### Saving the model

# In[ ]:


classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json)

classifier.save_weights("weights.h5")


# ## Building an SOM

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scale_object = MinMaxScaler(feature_range = (0,1))
X_som = scale_object.fit_transform(X)


# In[ ]:


from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 340, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X_som)
som.train_random(data = X_som, num_iteration = 100)


# In[ ]:


from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X_som):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()


# ## Loading the saved model to predict test results

# In[ ]:


test_set = pd.read_csv("equip_failures_test_set.csv")

for key in test_set:
    if "sensor" in key:
        test_set[key + "_na"] = [1 if test_set[key][i] == "na" else 0 for i in range(len(test_set[key]))]


# In[ ]:


new_X = test_set.iloc[:, 1:].values
new_X = np.where(new_X == "na", 0, new_X)
new_X = sc.fit_transform(new_X)


# In[ ]:


from keras.models import model_from_json
json_file = open("classifier.json", "r")
loaded_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_json)
classifier.load_weights("weights.h5")


# In[ ]:


new_predictions = classifier.predict(new_X)
new_predictions = new_predictions > 0.5
np.savetxt("test_results.txt", new_predictions, fmt = "%d")


# In[ ]:


with open("test_results.txt","r") as pred_file:
    fout = open("results_1.txt", "w")
    fout.write("id,target\n")
    for idx, val in enumerate(pred_file):
        fout.write(str(str(idx + 1) + "," + str(val)))
        


# In[ ]:


print(new_X.shape)
print(new_predictions[])

