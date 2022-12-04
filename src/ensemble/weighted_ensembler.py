import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from tqdm import tqdm


class WeightedModelsEnsembler:

    def __init__(self, model_list, target_columns, n_retries=1):
        self.model_list = model_list
        self.target_columns = target_columns

        self.weights = {col: {} for col in self.target_columns}
        self.scores = {col: 1 for col in self.target_columns}
        self.best_run_feature_importance = {model_name: 0 for model_name in model_list}
        self.all_runs_feature_importance = {model_name: 0 for model_name in model_list}
        self.mean_score = 1

        self.dataframe = None

        self.n_retries = n_retries

    def get_score(self, x0, column):
        predictions = self.dataframe[self.target_columns].copy()
        predictions[self.target_columns] = 0

        for fn, w in zip(self.model_list, x0):
            predictions[column] += self.dataframe['pred_' + column + '_' + fn] * w

        predictions[column] = predictions[column].clip(1, 5)
        score = mean_squared_error(self.dataframe[column].values, predictions[column].values, squared=False)
        return score

    def fit(self, X, options=None, bounds=None, method='SLSQP'):
        self.dataframe = X.copy()

        for _ in range(self.n_retries):
            all_columns_weights = {}
            all_columns_scores = {}

            for column in self.target_columns:
                initial_weights = np.random.dirichlet(np.ones(len(self.model_list)), size=1)[0]

                result = minimize(self.get_score,
                                  x0=initial_weights,
                                  args=(column,),
                                  options=options,
                                  bounds=bounds,
                                  method=method, )

                column_weights = {model: w for model, w in zip(self.model_list, result.x)}
                column_score = result.fun

                all_columns_weights[column] = column_weights
                all_columns_scores[column] = all_columns_scores

                if column_score < self.scores[column]:
                    self.scores[column] = column_score
                    self.weights[column] = column_weights

            feat_imp = self.get_feature_importance(all_columns_weights)
            for model_name in self.model_list:
                curr_imp = self.all_runs_feature_importance.get(model_name, 0)
                curr_imp += feat_imp[model_name] / self.n_retries
                self.all_runs_feature_importance[model_name] = curr_imp

        self.mean_score = np.mean(list(self.scores.values()))
        self.best_run_feature_importance = self.get_feature_importance(self.weights)
        self.all_runs_feature_importance = {imp[0]: imp[1] for imp in sorted(self.all_runs_feature_importance.items(), key=lambda x: x[1])}
        return self

    def predict(self, X):

        predictions = X.copy()
        predictions[self.target_columns] = 0

        for column in self.target_columns:
            column_weights = self.weights[column]
            for model_name, weight in column_weights.items():
                predictions[column] += predictions['pred_'+column+'_'+model_name] * weight

        predictions[self.target_columns] = predictions[self.target_columns].clip(1, 5)
        return predictions

    def get_weights_from_cv(self, submissions_cv_scores):
        weights_sum = {}
        weights = {model: {} for model in self.model_list}
        for model in self.model_list:
            model_scores = submissions_cv_scores[model]
            for col, score in model_scores.items():
                weight = 1 / score
                curr_sum = weights_sum.get(col, 0)
                curr_sum += weight
                weights_sum[col] = curr_sum

                weights[model][col] = weight

        normalized_weights = {model: {} for model in self.model_list}
        for model, w in weights.items():
            for col, weight in w.items():
                normalized_weights[model][col] = weight / weights_sum[col]
        return normalized_weights

    def get_feature_importance(self, weights):
        feature_importance = {}
        for col in self.target_columns:
            w_ = weights[col]
            for fn, w in w_.items():
                s = feature_importance.get(fn, 0)
                s += w
                feature_importance[fn] = s

        feature_importance = {imp[0]: imp[1] for imp in sorted(feature_importance.items(), key=lambda x: x[1])}
        return feature_importance
