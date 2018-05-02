
# coding: utf-8

# # Initial Data Exploration
# 
# Meant to provide some elementary visualisation, and to prepare for cleaning prior to using ML algos. 
# 
# *Author*: Jordan Giebas

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Metrics
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# ML Algos
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA


# ## Summary
# 
# **Complete Data**
# <ul>
#     <li> Our **complete** data has 762,678 rows and 61 features </li>
#     <li> There are 3736 **unique** bond ids, in the **complete** set. </li>
#     <li> There are 439351 `nan` values in our **complete** data </li>
#     <li> The correlation between `bond_id` and `trade_price` is $-0.20134$ </li>
#     <li> There are some `bond_id`s without 25 instances </li>
#     <ul>
#         <li> 1114 `bond_id`s dropped due to less than 25 instances </li>
#         <li> 12085 total rows insufficient data for ARMA </li>
#     </ul>
# </ul>
# 
# **Training Data**
# <ul>
#     <li>513582 observations</li>
#     <li>61 features</li>
# </ul>
# 
# **Test1 Data**
# <ul>
#     <li>110272 observations</li>
#     <li>61 features</li>
# </ul>
# 
# **Test2 Data**
# <ul>
#     <li>513582 observations</li>
#     <li>111468 features</li>
# </ul>
# 
# 
# **Variables**
# <ul>
#     <li> `complete_data`: (all data) dataframe from 'train.csv' </li>
#     <li> `bond_ids`: np.array of all bond_ids </li>
#     <li> `bond_id_dict`: dictionary mapping `bond_id` to that `bond_id`s number of occurrences </li>
#     <li> `bond_id_df`: same as above, but in `pd.DataFrame` form</li>
#     <li> `train_noNA`: `complete_data` with all `nans` dropped </li>
#     <li> `bond_ids_noNA`: bond ids for `train_noNA` dataframe</li>
#     <li> `bond_id_dict_noNA`: dictionary mapping `bond_ids_noNA` to number of occurrences</li>
#     <li> `bondID_gt25_noNA`: dictionary mapping `bond_ids_noNA` to number of instances, only if there are more than 25 instances </li>
#     <li> `clean_data`: dataframe w/ no `nan`s and `bond_id`s have more than 25 instances</li>
#     <li> `train_df`: dataframe containing first 70% of `clean_data`, stratified by `bond_id` </li>
#     <li> `test1_df`: dataframe containing middle 15% of `clean_data`, stratified by `bond_id` </li>
#     <li> `test2_df`: dataframe containing last 15% of `clean_data`, stratified by `bond_id` </li>
# </ul>

# In[2]:


complete_data = pd.read_csv('../shared_files/train.csv')
complete_data.shape


# **ORIGINAL TRAIN DF** <br>
# some scrubbing:

# In[3]:


bond_ids = np.array(complete_data.bond_id).tolist()
len(set(bond_ids))


# In[4]:


null_arr = [complete_data[col].isnull().sum() for col in complete_data.columns]
sum(null_arr)


# In[5]:


pearsonr(np.array(complete_data['bond_id']), np.array(complete_data['trade_price']))[0]


# In[6]:


bond_id_dict = {bond_id: bond_ids.count(bond_id) for bond_id in set(bond_ids)}


# In[7]:


series = pd.Series.from_array(bond_id_dict)
bond_id_df = pd.DataFrame()
bond_id_df['lengths'] = series
bond_ids_gt25samples = np.array(bond_id_df[bond_id_df.lengths > 25].index)
len(set(bond_ids)) - len(bond_ids_gt25samples.tolist())


# In[8]:


train_lt25_samples_dropped = complete_data[complete_data.bond_id.isin(bond_ids_gt25samples)]
len(complete_data) - len(train_lt25_samples_dropped)


# **Dropping all `nan`** Values, performing analysis on this dataframe

# In[9]:


# Drop all rows with NaN values
train_noNA = complete_data.dropna(axis=0)
# Define list of all bond_ids (there will be duplicates)
bond_ids_noNA = complete_data.bond_id.tolist()


# In[10]:


# Dictionary maps bond_id to number of occurrences for that id
bond_id_dict_noNA = {bond_id: bond_ids_noNA.count(bond_id) for bond_id in set(bond_ids_noNA)}
bondID_gt25_noNA = {bond_id: bond_ids_noNA.count(bond_id) for bond_id in set(bond_ids_noNA) if bond_ids_noNA.count(bond_id) > 25}


# In[11]:


len(bond_id_dict_noNA) - len(bondID_gt25_noNA)


# The above is the number of removals from not having enough instances. The keys in `bondID_gt25_noNA` are the bond ids that we must remove.

# In[12]:


# Histogram / scatter plots of the above.
fig, axs = plt.subplots(1,2)

fig.set_figheight(7.5)
fig.set_figwidth(15)
fig.align_ylabels

axs[0].hist(bond_ids_noNA, density=True, alpha=0.5, color='green')
axs[0].set_title("Histogram")

axs[1].scatter(bond_id_dict_noNA.keys(), bond_id_dict_noNA.values(), alpha=0.5)
axs[1].set_title("Scattered")


# Now all of the NAs are gone, we must determine which `bond_id`s have less than 10 instances.

# In[13]:


# Final Train: No NaN Values, and all bond_id instances > 25
clean_data = train_noNA[train_noNA.bond_id.isin(bondID_gt25_noNA.keys())]
clean_data.index = pd.RangeIndex(1, len(clean_data)+1)


# In[14]:


clean_data.shape


# The above is the number of rows that have been removed from that overall dataframe for not having a sufficient amount of `bond_id` instances.

# In[15]:


len(set(clean_data.bond_id.tolist()))


# After the removal of any `bond_id` without more than 25 instances, we are left with 2473 `bond_id`s (i.e. time-series since this is a 1-1 mapping)

# # Splitting into Training / Testing Sets
# 
# (Per 4/19/2018)<br>
# Since Kaggle didn't provide the response variable in the test data, we must split the training data up into a training and test set. Specifically, of the original data 70% will be our new training set, 15% will be one test set, and the last 15% will be left-out test set. <br>
# Per the discussion Lucas and I had w/ Max on 4/20 ;), the following method will be used to split the data <br>
# 
# For each `bondID`:
# <ol> 
#     <li> Define `df` as a slice of `final_train` such that `bond_id = bondID` </li>
#     <li> Define `N = len(df)` </li>
#     <li> Append `[: to the train_df_list)` </li>
#     <li> Append `[np.floor(0.7N)):np.floor(0.85N))` to the `test1_df_list` </li>
#     <li> Append `[np.floor(0.85N):)` to the `test2_df_list` </li>
# </ol>
# 
# Stack all dataframes on top of one another using `pd.concat(DataFrame_list)`
# 
# 
# **Note**: <br>
# Our `test2_df` will hold the most recent data, `test1_df` the next most recent, and the `train_df` the rest. <br>
# 
# Our procedure will then be to: <br>
# <ol>
#     <li> Train on `train_df` and predict against `test1_df` </li>
#     <li> Train on `train_df` and `test1_df` then predict against `tests2_df` </li>
# </ol>
# 
# This way predictions are in the future as opposed the random sampling idea from before. 

# In[16]:


# init dataframes
train_df_list = []; test1_df_list = []; test2_df_list = [];

# Populate dataframes
for bondID in sorted(list(set(clean_data.bond_id.tolist()))):
    
    df = clean_data[clean_data['bond_id'] == bondID]
    N  = len(df)
    
    train_df_list.append( df.iloc[:int(np.floor(0.7*N)), :] )
    test1_df_list.append( df.iloc[int(np.floor(0.7*N)):int(np.floor(0.85*N)), :] )
    test2_df_list.append( df.iloc[int(np.floor(0.85*N)):, :] )

train_df = pd.concat(train_df_list)
test1_df = pd.concat(test1_df_list)
test2_df = pd.concat(test2_df_list)


# In[17]:


print(len(train_df)/len(clean_data));print(len(test1_df)/len(clean_data)); print(len(test2_df)/len(clean_data))


# In[18]:


plt.plot(train_df[train_df['bond_id'] == 2].time_to_maturity.tolist())


# I wanted to be sure that the splitting worked. You can see that the proportions are roughly 70%, 15%, 15% so this is good. The plot above was to make sure for a given `bond_id`, that the rows were inserted correctly- this is confirmed since we see the time to maturity monotonically decreasing. 

# ## Loss Measure: *WEPS*

# In[19]:


"""
Input Params:
    weights (np.array): weight given to every entry / trade
    y_true  (np.array): true y values 
    y_pred  (np.array): predicted y values 
    
Return Params:
    WEPS loss measure as defined in paper
"""
def WEPS(weights, y_true, y_pred):
    return sum(np.multiply(weights, np.absolute(y_true-y_pred))) / sum(weights)


# ## Algorithms
# Since a lot of these are supervised learning methods, `train_df`, `test1_df`, and `test2_df` are split up below into their feature and response variables <br>
# 
# **Note/Question**: Do we want to use `%timeit` to get run times like the paper did? I think we'll be able to run most algorithms, per 4/20 I'm only unsure about NNs

# In[20]:


y_train = train_df.trade_price; X_train = train_df.drop('trade_price', axis=1)
y_test1 = test1_df.trade_price; X_test1 = test1_df.drop('trade_price', axis=1)
y_test2 = test2_df.trade_price; X_test2 = test2_df.drop('trade_price', axis=1)


# The below dataframes are the 85% of the data as training (i.e. X_train stacked on top of X_test1 in one dataframe)

# In[21]:


y_train2 = pd.concat([y_train,y_test1]); X_train2 = pd.concat([X_train, X_test1])


# ### OLS (trivial)
# For this, should we drop the weights in the feature space and then use them for WLS? 
# 
# #### Use `X_train` to fit model

# In[22]:


# Fit the model
lreg = LinearRegression().fit(X_train, y_train) 


# In[23]:


# Test Prediction / MSE
lreg_ypred_test = lreg.predict(X_test1)
lreg_mse_test  = mean_squared_error(lreg_ypred_test, y_test1) 

# Train Prediction / MSE
lreg_ypred_train = lreg.predict(X_train)
lreg_mse_train  = mean_squared_error(lreg_ypred_train, y_train)

# WEPS Loss Measure
lreg_WEPS_test = WEPS(X_test1.weight, y_test1, lreg_ypred_test)
lreg_WEPS_train = WEPS(X_train.weight, y_train, lreg_ypred_train)


# In[24]:


# Output error
error_mat = np.array([[lreg_mse_train, lreg_mse_test], [lreg_WEPS_train, lreg_WEPS_test]])
pd.DataFrame(error_mat, index=['MSE', 'WEPS'], columns=['Train', 'Test'])


# #### Use `X_train` and `X_test1` to fit model

# In[25]:


# Fit the model
lreg2 = LinearRegression().fit(X_train2, y_train2) 


# In[26]:


# Test Prediction / MSE
lreg2_ypred_test = lreg2.predict(X_test2)
lreg2_mse_test  = mean_squared_error(lreg2_ypred_test, y_test2) 

# Train Prediction / MSE
lreg2_ypred_train = lreg2.predict(X_train2)
lreg2_mse_train  = mean_squared_error(lreg2_ypred_train, y_train2)

# WEPS Loss Measure
lreg2_WEPS_test = WEPS(X_test2.weight, y_test2, lreg2_ypred_test)
lreg2_WEPS_train = WEPS(X_train2.weight, y_train2, lreg2_ypred_train)


# In[27]:


# Output error
error_mat = np.array([[lreg2_mse_train, lreg2_mse_test], [lreg2_WEPS_train, lreg2_WEPS_test]])
pd.DataFrame(error_mat, index=['MSE', 'WEPS'], columns=['Train', 'Test'])


# ### WLS 
# Use the `weights` column of each dataframe as the weights parameter in `LinearRegression().fit()` <br>
# Refer to: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# In[28]:


ws = X_train.weight
X_train_no_weights = X_train.drop('weight', axis=1)


# In[29]:


# Fit the model
wls_reg = LinearRegression().fit(X_train_no_weights, y_train, sample_weight=ws) 


# In[30]:


# Test Prediction / MSE
wls_reg_ypred_test = wls_reg.predict(X_test1.drop('weight', axis=1))
wls_reg_mse_test  = mean_squared_error(wls_reg_ypred_test, y_test1) 

# Train Prediction / MSE
wls_reg_ypred_train = wls_reg.predict(X_train_no_weights)
wls_reg_mse_train  = mean_squared_error(wls_reg_ypred_train, y_train)

# WEPS Loss Measure
wls_reg_WEPS_test = WEPS(X_test1.weight, y_test1, lreg_ypred_test)
wls_reg_WEPS_train = WEPS(X_train.weight, y_train, lreg_ypred_train)


# In[31]:


# Output error
error_mat = np.array([[wls_reg_mse_train, wls_reg_mse_test], [wls_reg_WEPS_train, wls_reg_WEPS_test]])
pd.DataFrame(error_mat, index=['MSE', 'WEPS'], columns=['Train', 'Test'])


# ### PCA (dimension reduction)

# In[32]:


from sklearn.decomposition import PCA
pca_obj = PCA().fit(train_df)
transformed = pca_obj.transform(train_df)


# In[33]:


sum(np.cumsum(pca_obj.explained_variance_ratio_) < 0.99999)


# This is on point, confirmed w/ @Skander's and the paper

# ### Random Forest
# **Note** Using `n_jobs` indicates with how many cores you would like to parallelize the code. Using `n_jobs=-1` sets it the number of cores on your Machine. You can check this by entering into the terminal (or bash, I think): <br><br>
# 
# <center>`sysctl hw.ncpu`</center>

# #### 50 Regression Trees

# In[34]:


get_ipython().run_cell_magic('timeit', '', 'rfr_50 = RandomForestRegressor(n_estimators=50, max_depth=2, n_jobs=-1).fit(X_train, y_train)')


# Let's be careful about this `timeit` business....

# In[35]:


rfr_50 = RandomForestRegressor(n_estimators=50, max_depth=2, n_jobs=-1).fit(X_train, y_train)


# In[36]:


rfr50_ypred_test  = rfr_50.predict(X_test1)
rfr50_ypred_train = rfr_50.predict(X_train)

rfr50_mse_test  = mean_squared_error(rfr50_ypred_test, y_test1)
rfr50_WEPS_test = WEPS(X_test1.weight, y_test1, rfr50_ypred_test)

rfr50_mse_train  = mean_squared_error(rfr50_ypred_train, y_train)
rfr50_WEPS_train = WEPS(X_train.weight, y_train, rfr50_ypred_train)


# In[37]:


# Output error
error_mat = np.array([[rfr50_mse_train, rfr50_mse_test], [rfr50_WEPS_train, rfr50_WEPS_test]])
pd.DataFrame(error_mat, index=['MSE', 'WEPS'], columns=['Train', 'Test'])


# #### 100 Regression Trees

# In[38]:


rfr_100 = RandomForestRegressor(n_estimators=100, max_depth=2, n_jobs=-1).fit(X_train, y_train)


# In[39]:


rfr100_ypred_test  = rfr_100.predict(X_test1)
rfr100_ypred_train = rfr_100.predict(X_train)

rfr100_mse_test  = mean_squared_error(rfr100_ypred_test, y_test1)
rfr100_WEPS_test = WEPS(X_test1.weight, y_test1, rfr100_ypred_test)

rfr100_mse_train  = mean_squared_error(rfr100_ypred_train, y_train)
rfr100_WEPS_train = WEPS(X_train.weight, y_train, rfr100_ypred_train)


# In[40]:


# Output error
error_mat = np.array([[rfr100_mse_train, rfr100_mse_test], [rfr100_WEPS_train, rfr100_WEPS_test]])
pd.DataFrame(error_mat, index=['MSE', 'WEPS'], columns=['Train', 'Test'])


# ### Neural Networks

# #### 5 Neurons

# In[154]:


import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


# In[155]:


# Define the neural net class
class Net(nn.Module):
    def __init__(self, input_dimension, hidden_size, target_dimension):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dimension, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, target_dimension)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# In[185]:


# Let's play around w/ the hidden layer dimension
hidden_ = 5
learning_rate = 0.001 ### This was chosen as the tutorial, a lot of literature on this....

# Convert training/testing data set into a torch tensors
xtorch_train = torch.from_numpy(np.asmatrix(X_train))
xtorch_test = torch.from_numpy(np.asmatrix(X_test1))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=xtorch_train, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=xtorch_test, 
                                          shuffle=False)


# In[157]:


net = Net(X_train.shape[1], hidden_, X_train.shape[0])


# In[158]:


# Loss and Optimizer
criterion = nn.MSELoss()    ### Let's do MSE loss for now, perhaps nn.L1Loss() later
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


# In[159]:


ytorch_train1 = Variable(torch.from_numpy(np.array(y_train, dtype=np.float64))).float()

# Fit the model
for i, elm in enumerate(train_loader):
    
    # tensor to autograd.Variable
    tmp_x = Variable(elm).float()
    
    # Forward + Backward + Optimize
    optimizer.zero_grad() 
    y_pred = net(tmp_x)     
    
    loss = criterion(y_pred, ytorch_test1)
    loss.backward()
    optimizer.step()


# In[210]:


# Test the Model
for i, elm in enumerate(test_loader):
    
    tmp_x = Variable(elm).float()
    y_pred = net(tmp_x)
    
    #print(len(y_pred.data.numpy()[0]))
    print(len(tmp_x.data.numpy()[0]))
    break


# In[172]:


# Save the model
torch.save(net.state_dict(), 'model.pkl')


# In[206]:


xtorch_test


# In[1]:


import torch


# In[2]:


torch.cuda.is_available()

