
"""
This is the class to standardize data

"""

from sklearn.preprocessing import StandardScaler

class standardize_data():

    def __init__(self, data):
        self.init_data = data
        self._stand_data = []
        self.standard_it(data)

    def standard_it(self, data):
        self._stand_data = StandardScaler().fit_transform(data)

    def get_stand_data(self):
        a = self._stand_data
        return a




