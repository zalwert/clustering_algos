"""

This is the main class of the script
There are nine clustering algorithms utilized:

clustering_algorithms = [

                        'MiniBatchKMeans',
                        'AffinityPropagation',
                        'MeanShift',
                        'SpectralClustering',
                        'Ward',
                        'AgglomerativeClustering',
                        'DBSCAN',
                        'Birch',
                        'GaussianMixture'

                       ]

code based on:
https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

"""
import pandas as pd
import numpy as np
import time
from PCA_class import pca_data
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from Standard_class import standardize_data
from settings import default_settings

#Ignoring warnings as there are int64 to float64 issues
warnings.filterwarnings('ignore')

print("Loading data")
excel_path = "/Users/tanalytics1/Desktop/AI artykuły/dataWSE.xlsx"
test_data_all = pd.read_excel(excel_path, index_col=0, sheet_name='data')

print("Data loaded correctly")
test_data = test_data_all.loc[:, test_data_all.columns != 'Industry']
test_data_y = test_data_all.loc[:, test_data_all.columns == 'Industry']
test_data_pd = test_data

print("SNS pair plotting")
#Plot pairplot and save in the local directory
sns_plot = sns.pairplot(test_data_pd)
sns_plot.savefig("SNSpairplot.png")

print("Standardize data")
#Standardization of data
std_cls = standardize_data(test_data_pd)
std_data = std_cls.get_stand_data()

print("PCA transformation algorithm")
#PCA transformation
pca_cls = pca_data(test_data_pd, n_components = 5)
pca_dataset = pca_cls.get_pca_data()
pca_comp = pca_cls.get_pca_data_components()

plt.figure(figsize=(9 * 2 + 3, 10.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
plot_num = 1
default_base = default_settings

datasets = [
    ((test_data, test_data_y), {'damping': .75, 'preference': -250,
                     'quantile': .27, 'n_clusters': 4})]

for i_dataset, (dataset, algo_params) in enumerate(datasets):

    print("Started clustering algorithms")
    params = default_base.copy()
    params.update(algo_params)
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward',connectivity=connectivity)
    spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    )

    for name, algorithm in clustering_algorithms:

        t0 = time.time()
        #Fitting data into algorithms
        print("Fitting data into algorithms", name)
        algorithm.fit(X)
        t1 = time.time()
        print("Time taken ", t1)

        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(4, len(clustering_algorithms), plot_num)

        if i_dataset == 0:
            plt.title(name, size=10)

        colors = np.array(list(islice(cycle(['#F15D57', '#2878BD', '#38354B',
                                             '#94C83D', '#EFB032', '#984ea3',
                                             '#A26769', '#5A716A', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.97, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plot_num += 1

plt.savefig('clusters.png')




