
import pandas as pd 
import numpy as np
wine = pd.read_csv("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\\Assignments\\PCA\\wine.csv")
wine.describe()
wine.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Considering only numerical data 

# Normalizing the numerical data 
def norm_func(i):
    x = (i-i.min())	/(i.max()-i.min())
    return (x)

df_norm = norm_func(wine)
df_norm.columns


pca = PCA(n_components = 6)
pca_values = pca.fit_transform(df_norm)


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:2:14]
plt.scatter(x,y,color=["red","blue"])

from mpl_toolkits.mplot3d import Axes3D
Axes3D.scatter(np.array(x),np.array(y),np.array(z),c=["green","blue","red"])


################### Hierarchical Clustering  ##########################
new_df = pd.DataFrame(pca_values[:,0:14])
from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 


#p = np.array(df_norm) # converting into numpy array format 

z = linkage(new_df, method="complete",metric="euclidean")

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
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(new_df) 


cluster_labels=pd.Series(h_complete.labels_)

wine['clust']=cluster_labels # creating a  new column and assigning it to new column 
wine.columns
wine = wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine.head()

####################### kMeans Clustering####################
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4) 
model.fit(new_df)


model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine['clust']=md # creating a  new column and assigning it to new column 
new_df.head()
wine = wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]


wine.iloc[:,1:15].groupby(wine.clust).mean()
