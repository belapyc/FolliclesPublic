from modelling.helper import cycles_to_df_for_binpred, df_to_long_xy


class RFEnsemble:

    def __init__(self, models):
        self.models = models

    def prepare_data_for_pred(self, cycles, num_scans_to_use=None):
        df = cycles_to_df_for_binpred(cycles, bin_size=3, num_scans_to_use=num_scans_to_use)
        # print('df in prepare_data_for_pred:')
        # print(df)
        df = df.drop(columns=['afc', 'amh'])
        X, _ = df_to_long_xy(df, bin_size=3, drop_id=False, num_scans_to_use=num_scans_to_use)
        return X

    def predict(self, X):
        bins = self.models['bins']
        # bins_str = [str(bins[i]) + '-' + str(bins[i + 1]) for i in range(len(bins) - 1)]
        bins_str = bins
        X_for_rf = X.drop(columns='id').copy()
        for bin in bins_str:
            X[bin + '_pred'] = self.models[bin].predict(X_for_rf)

        return X