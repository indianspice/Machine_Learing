
# coding: utf-8

# ## 1. Import Python libraries


# used to change filepaths
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

# import Image from PIL
from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# import accuracy_score from sklearn's metrics module
from sklearn.metrics import roc_curve, auc


# In[74]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task_1():\n    assert 'Image' in globals(), \\\n    'Did you forget to import `Image` from `PIL`?'\n    \ndef test_task_2():\n    assert 'train_test_split' in globals(), \\\n    'Did you forget to import `train_test_split` from `sklearn.model_selection`?'\n    \ndef test_task_3():\n    assert 'SVC' in globals(), \\\n    'Did you forget to import `SVC` from `sklearn.svm`?'\n    \ndef test_task_4():\n    assert 'accuracy_score' in globals(), \\\n    'Did you forget to import `accuracy_score` from `sklearn.metrics`?'")


# ## 2. Display image of each bee type

# In[85]:


# load the labels using pandas
labels = pd.read_csv("datasets/labels.csv", index_col=0)

# show the first five rows of the dataframe using head
labels.head()

def get_image(row_id, root="datasets/"):
    """
    Converts an image number into the file path where the image is located, 
    opens the image, and returns the image as a numpy array.
    """
    filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)

# subset the dataframe to just Apis (genus is 0.0) get the value of the sixth item in the index
apis_row = labels[labels.genus == 0.0].index[5]

# show the corresponding image of an Apis
plt.imshow(get_image(apis_row))
plt.show()

# subset the dataframe to just Bombus (genus is 1.0) get the value of the sixth item in the index
bombus_row = labels[labels.genus == 1.0].index[6]

# show the corresponding image of a Bombus
plt.imshow(get_image(bombus_row))
plt.show()


# In[76]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task2_0():\n    assert (bombus_row == 1934), \\\n    'Did you get the sixth row of the index of the subsetted dataframe (labels[labels.genus == 0.0])?'")


# ## 3. Image manipulation with rgb2grey



# load a bombus image using our get_image function and bombus_row from the previous cell
bombus = get_image(bombus_row)

# print the shape of the bombus image
print('Color bombus image has shape: ', bombus)

# convert the bombus image to greyscale
grey_bombus = rgb2grey(bombus)

# show the greyscale image
plt.imshow(grey_bombus, cmap=mpl.cm.gray)

# greyscale bombus image only has one channel
print('Greyscale bombus image has shape: ', grey_bombus)


# In[78]:


get_ipython().run_cell_magic('nose', '', "import numpy\n\ndef test_task3_0():\n    assert 'bombus' in globals() and bombus.shape == (100, 100, 3), \\\n    'Did you load the image corresponding to `bombus_row` using the `get_image` function and assign it to `bombus`?'\n    \ndef test_task3_1():\n    assert grey_bombus.shape == (100, 100) and grey_bombus.max() <= 1, \\\n    'Did you convert `bombus` to greyscale using `rgb2grey`?'")


# ## 4. Histogram of oriented gradients



# run HOG using our greyscale bombus image
hog_features, hog_image = hog(grey_bombus,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16))

# show our hog_image with a grey colormap
plt.imshow(hog_image, cmap=mpl.cm.gray)


# In[80]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task4_0():\n    assert all(hog_image[0] == np.array([0] * 100)), \\\n    'Did you call `hog` on `grey_bombus`?'\n    \ndef test_task4_1():\n    assert '_' in globals() and isinstance(globals()['_'], mpl.image.AxesImage), \\\n    'Did you forget to call `plt.imshow` on `hog_image`?'")


# ## 5. Create image features and flatten into a single row



def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack(color_features)
    return flat_features

bombus_features = create_features(bombus)

# print shape of bombus_features
print(bombus_features)


# In[82]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task5_0():\n    assert bombus_features[29999] == 118.0 and round(bombus_features[30000], 3) == 0.053, \\\n    'Did you pass color_features and hog_featuers into the np.hstack function in this order?'\n    \ndef test_task5_1():\n    assert bombus_features.shape == (31296,), \\\n    '`bombus_features` does not have the correct shape. Did you setup the `create_features` function properly?'")


# ## 6. Loop over images to preprocess



def create_feature_matrix(label_dataframe):
    features_list = []
    
    for img_id in label_dataframe.index:
        # load image
        img = get_image(img_id)
        # get features for image
        image_features = create_features(img)
        features_list.append(image_features)
        
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix

# run create_feature_matrix on our dataframe of images
feature_matrix = create_feature_matrix(labels)


# In[ ]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task6_0():\n    assert feature_matrix[0, -1] != feature_matrix[1, -1], \\\n    'Did you call `create_features` on `img`?'\n    \ndef test_task6_1():\n    assert feature_matrix.shape == (500, 31296), \\\n    'Did you call `create_feature_matrix` on the dataframe `labels`?'")


# ## 7. Scale feature matrix + PCA



# get shape of feature matrix
print('Feature matrix shape is: ', feature_matrix.shape)

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
bees_stand = ss.fit_transform(feature_matrix)

pca = PCA(n_components=500)
# use fit_transform to run PCA on our standardized matrix
bees_pca = ss.fit_transform(bees_stand)
# look at new shape
print('PCA matrix shape is: ', bees_pca.shape)


# In[ ]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task7_0():\n    assert round(bees_stand[0, 0], 3) == 0.097, \\\n    'Did you pass in `feature_matrix` to `ss.fit_transform`?'\n    \ndef test_task7_1():\n    assert round(bees_pca[0, 0], 3) == 19.335, \\\n    'Did you pass in `bees_stand` to `pca.fit_transform`?'\n    \ndef test_task7_2():\n    assert bees_pca.shape == (500, 500), \\\n    'Did you pass in `bees_stand` to `pca.fit_transform`?'")


# ## 8. Split into train and test sets



X = pd.DataFrame(bees_pca)
y = pd.Series(labels.genus.values)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.3,
                                                    random_state=1234123)

# look at the distrubution of labels in the train set
pd.Series(y_train).value_counts()


# In[ ]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task8_0():\n    assert '_' in globals() and isinstance(globals()['_'], pd.Series), \\\n    'Did you forget to forget to look at the distribution of labels using `pd.Series(y_train).value_counts()`?'\n    \ndef test_task8_1():\n    assert X_train.shape == (350, 500), \\\n    'Did you pass `bees_pca` as X into train_test_split?'\n                                     \ndef test_task8_2():\n    assert y_train.shape == (350,) and (np.unique(y_train) == [0., 1.,]).all(), \\\n    'Did you pass `labels.genus.values` as y into train_test_split?'")


# ## 9. Train model


# define support vector classifier
svm = SVC(kernel='linear', probability=True, random_state=42)

# fit model
svm.fit(X_train, y_train)


# In[ ]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task9_0():\n    assert svm.kernel == 'linear' and svm.probability == True, \\\n    'Did you assign define an SVC with a linear kernel and set probability equal to True?'")


# ## 10. Score model



# generate predictions
y_pred = svm.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)


# In[ ]:


get_ipython().run_cell_magic('nose', '', "    \ndef test_task9_1():\n    assert pd.Series(y_pred).value_counts()[0] == 79, \\\n    'Did you generate predictions using `svm.predict(X_test)`?'\n    \ndef test_task9_2():\n    assert round(accuracy, 2) == 0.68, \\\n    'Did you calculate accuracy using `accuracy_score(y_test, y_pred)`?'")


# ## 11. ROC curve + AUC


# predict probabilities for X_test using predict_proba
probabilities = svm.predict_proba(X_test)

# select the probabilities for label 1.0
y_proba = probabilities[:, 1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate');


# In[ ]:


get_ipython().run_cell_magic('nose', '', "\ndef test_task10_0():\n    assert round(y_proba[0], 1) == .5 , \\\n    'Did you define probabilities using `svm.predict_proba(X_test) and y_proba using `probabilityes[:, 1]?'\n\ndef test_task10_1():\n    assert false_positive_rate.shape == (70,), \\\n    'Did you pass `y_test, y_proba, pos_label=1` into roc_curve?'\n    \ndef test_task10_2():\n    assert round(roc_auc, 2) == .74, \\\n    'Did you calculate the roc_auc properly?'\n    \ndef test_task10_3():\n    x, y = roc_plot[0].get_data()\n    assert x[10] == 0.08 and round(y[10], 2) == 0.28, \\\n    'Did you plot the false positive rate on the x axis and the true positive rate on the y axis?'")

