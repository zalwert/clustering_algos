"""
This is the class to decompose datasets into
principal components/elements

Linear dimensionality reduction using Singular
Value Decomposition of the data to project it
to a lower dimensional space.

"""

from sklearn.decomposition import PCA

class pca_data():

    def __init__(self, data, n_components = 6):
        self.init_data = data
        self._pca_data = []
        self.pca = PCA(n_components)
        self.pca_it(data)

    def pca_it(self, data):
        self.pca.fit_transform(data)

    def get_pca_data(self):
        a = self.pca.explained_variance_ratio_
        return a

    def get_pca_data_components(self):
        a = self.pca.components_
        return a
