# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:09:27 2020

@author: HP
"""

import pandas as pd
import matplotlib.pylab as plt 
crime = pd.read_csv("C:\\Users\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\Clustering\\crime_data.csv")

crime.rename(columns={crime.columns[0]:'city'}, inplace=True)
crime.columns
#pd.get_dummies(crime, columns=['city'], drop_first=True)
# Normalization function 

def norm_func(i):
    x = (i-i.min())	/(i.max()-i.min())
    return (x)

# alternative normalization function 

#def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

help(linkage)

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)

crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.head()

# getting aggregate mean of each cluster
crime.iloc[:,2:].groupby(crime.clust).median()

# creating a csv file 
crime.to_csv("crimee.csv",encoding="utf-8")

