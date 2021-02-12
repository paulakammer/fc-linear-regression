import numpy as np
from sklearn.linear_model import LinearRegression


class LinearRegressionClient:
    model = LinearRegression()
    X = None
    y = None
    X_test = None
    y_test = None
    X_offset_local = None
    y_offset_local = None
    X_scale_local = None
    X_offset_global = None
    y_offset_global = None
    X_scale_global = None

    def set_coefs(self, coef):
        self.model.coef_ = coef
        self.model._set_intercept(self.X_offset_global, self.y_offset_global, self.X_scale_global)

    def set_global_offsets(self, aggregated_preprocessing):
        self.X_offset_global = aggregated_preprocessing[0]
        self.y_offset_global = aggregated_preprocessing[1]
        self.X_scale_global = aggregated_preprocessing[2]

        self.X -= self.X_offset_global
        self.y -= self.y_offset_global

    def local_preprocessing(self):
        accept_sparse = False if self.model.positive else ['csr', 'csc', 'coo']

        self.X, self.y = self.model._validate_data(self.X, self.y, accept_sparse=accept_sparse, y_numeric=True,
                                                   multi_output=True)

        # if regr.sample_weight is not None:
        #    sample_weight = regr._check_sample_weight(sample_weight, X,dtype=X.dtype)
        _, _, self.X_offset_local, self.y_offset_local, self.X_scale_local = self.model._preprocess_data(
            self.X, self.y, fit_intercept=self.model.fit_intercept, normalize=False,
            copy=self.model.copy_X, sample_weight=None, return_mean=True)

    def local_computation(self):
        XT_X_matrix = np.dot(self.X.T, self.X)
        XT_y_vector = np.dot(self.X.T, self.y)

        return XT_X_matrix, XT_y_vector
