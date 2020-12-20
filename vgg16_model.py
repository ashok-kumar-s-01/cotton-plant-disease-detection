#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.resnet152V2 import ResNet152V2
from keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[2]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = r'C:\Users\prash\Desktop\data\cotton plant leaf disease\train2'
valid_path = r'C:\Users\prash\Desktop\data\cotton plant leaf disease\test2'


# In[3]:


import tensorflow
vgg16 =tensorflow.keras.applications.VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[4]:


for layer in vgg16.layers:
    layer.trainable = False


# In[5]:


folders = glob(r'C:\Users\prash\Desktop\data\cotton plant leaf disease\train2\*')


# In[6]:


x = Flatten()(vgg16.output)


# In[7]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg16.input, outputs=prediction)


# In[8]:


model.summary()


# In[9]:


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[10]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[11]:


training_set = train_datagen.flow_from_directory(r'C:\Users\prash\Desktop\data\cotton plant leaf disease\train2',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[12]:


test_set = test_datagen.flow_from_directory(r'C:\Users\prash\Desktop\data\cotton plant leaf disease\test2',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[13]:


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[14]:


plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[15]:


from tensorflow.keras.models import load_model

model.save(r'C:\Users\prash\Desktop\data\cotton plant leaf disease\model_vgg16_2.h5')


# In[16]:


y_pred = model.predict(test_set)


# In[17]:


y_pred


# In[18]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[19]:


y_pred


# In[ ]:





# In[20]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[21]:


model=load_model(r'C:\Users\prash\Desktop\data\cotton plant leaf disease\model_vgg16_2.h5')


# In[23]:


model


# In[ ]:




