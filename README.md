# mallSegmentaionKMeanClustering
<h1>Mall Customer Segmentation (K-Means Clustering)</h1>

<h2>Purpose</h2>
To determine the optimal number of clusters for K-Means on Mall Customer Data.
<br />


<h2>Languages </h2>

- <b>Python</b> 
  

<h2>Environments Used </h2>

- <b> Google  Colab</b> 

<h2>Data Sources and Dataset Description </h2>

This study focuses on a data of about 200 mall customers. The data contains their gender, age , annual income in dollars and their respective spending score measured over 100
 <br/>

<p align="center">
<img src="https://imgur.com/lVBBXGa.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />

<h2>Checking our data to see if null values exist </h2>

<p align="center">
<img src="https://imgur.com/3RtHKvx.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />


<h2> Retreiving Annual Income on column 3 and Spending Score on coulmn 4 </h2>

 <b>  x = df.iloc[:, [3,4]].values </b> 

<p align="center">
<img src="https://imgur.com/VXSY4iS.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />

<h2>Performing Elbow Method to find optimal No. of Clusters</h2>

- <b> from sklearn.cluster import KMeans.</b>
- <b> NB: WSCC => WITHIN CLUSTER SUMMER SQUARES.</b>

<p align="center">
<img src="https://imgur.com/umqb4rt.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />

<h2>PLOT THE ELBOW GRAPH TO SHOW WHICH CLUSTERS WHICH HAVE THE MINIMUM VALUES</h2>
<p align="center">
<img src="https://imgur.com/5dTslzu.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />

<h2>ELBOW GRAPH</h2>
<p align="center">
<img src="https://imgur.com/3onZlGs.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />


<h2>observation</h2>
- <b> 1</b>
- <b> 2</b>

<h2>Training a model using unsupervised learning algorithm (K-Means)</h2>
Initializing our K-means model with selected optimal no. of clusters <br/>
- <b> We will plot clusters and gain intuitions regarding our customers</b>
<p align="center">
<img src="https://imgur.com/Q3P3SQG.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />

<h2> Cluster Visualization </h2>
Script
<p align="center">
<img src="https://imgur.com/p2k8YNp.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />


<p align="center">
<img src="https://imgur.com/6BvG7a6.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<h2>OBSERVATION </h2>
- <b> 1</b>
- <b> 2</b>
