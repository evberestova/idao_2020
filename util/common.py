from collections import Counter

import numpy as np
import pandas as pd


DF_FEATURES = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']
DF_TARGETS = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']


def _check_df(df, additional_columns=None):
    additional_columns = additional_columns or []
    df_columns = set(df.columns)
    for val in additional_columns + ['sat_id'] + DF_FEATURES:
        assert val in df_columns, f'DataFrame must contain "{val}" column. {df_columns}'


def feature_to_target(feature):
    return feature[:-4] # drop '_sim' suffix


def target_to_feature(target):
    assert not target.endswith('_sim'), '{} is a feature already'.format(target)
    return target + '_sim'


def load_data(train_file, test_file, train_anomaly_threshold=10, test_anomaly_threshold=None):
    def _sort_df(df):
        df.sort_values(by=['sat_id', 'epoch'], ascending=[True, True], inplace=True)
    
    def _find_anomalies(df, anomaly_threshold):
        epoch_diff = df.epoch.values[1:] - df.epoch.values[:-1]
        diff_counter = Counter(epoch_diff)
        anomaly_mask = np.full(df.shape[0] - 1, True, dtype=np.bool)
        for diff, count in diff_counter.items():
            if count > anomaly_threshold:
                continue
            anomaly_mask = np.logical_xor(anomaly_mask, epoch_diff == diff)
        return np.hstack([[True], anomaly_mask])
    
    def _load_data_frame(path, is_train, anomaly_threshold):
        result = pd.read_csv(path, parse_dates=['epoch'])
        _check_df(result, ['id'] + DF_TARGETS if is_train else [])
        _sort_df(result)
        if anomaly_threshold is not None:
            anomalies = np.array([], dtype=np.bool)
            for _, df in result.groupby('sat_id'):
                sat_anomalies = _find_anomalies(df, anomaly_threshold)
                anomalies = np.hstack([anomalies, sat_anomalies])
            result = result[anomalies]
        result.reset_index(drop=True, inplace=True)
        return result
    
    return _load_data_frame(train_file, True, train_anomaly_threshold), \
           _load_data_frame(test_file, False, test_anomaly_threshold)


def calculate_score(df):
    def _calc_smape(lhs, rhs):
        return np.mean(np.abs((lhs - rhs) / (np.abs(lhs) + np.abs(rhs))))
    
    def _calc_score(smapes):
        return 100 * (1 - np.mean(smapes))
    
    _check_df(df, DF_TARGETS)
    per_sat_scores = []
    for _, data in df.groupby('sat_id'):
        smapes = _calc_smape(data[DF_FEATURES].values, data[DF_TARGETS].values)
        per_sat_scores.append(smapes)
    return _calc_score(per_sat_scores)
