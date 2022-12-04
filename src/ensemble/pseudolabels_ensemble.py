import pandas as pd
import os


def make_modelwise_submission_ensemble(pseudolabels_path, model_weights, output_dir, n_folds=5):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    model_paths = [os.path.join(pseudolabels_path, f'{model_id}_pseudolabels') for model_id in model_weights.keys()]

    df2 = pd.read_csv(os.path.join(pseudolabels_path, 'model2_pseudolabels', 'pseudolabels_fold0.csv'))
    df2 = df2.reset_index()

    target_columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    for fold in range(n_folds):
        ensemble = pd.read_csv(os.path.join(pseudolabels_path, 'model2_pseudolabels', 'pseudolabels_fold0.csv'))
        ensemble[target_columns] = 0

        for path, weight in zip(model_paths, model_weights.values()):
            df = pd.read_csv(os.path.join(path, f'pseudolabels_fold{fold}.csv'))

            if not (df.text_id.values[0] == '73D6F19E24BD' and df.text_id.values[-1] == '6D2AD3027292'):
                print(path, fold, ' not in right order, fixing...')

                df = pd.merge(df, df2[['text_id', 'index']], on=['text_id'], how='left')
                df = df.sort_values('index')
                df = df.reset_index(drop=True)
                df.drop(['index'], axis=1, inplace=True)

                if not (df.text_id.values[0] == '73D6F19E24BD' and df.text_id.values[-1] == '6D2AD3027292'):
                    print(path, fold, ' not fixed')

            ensemble[target_columns] += df[target_columns] * weight

        ensemble.to_csv(os.path.join(output_dir, f'pseudolabels_fold{fold}.csv'), index=False)
    return True


def make_columnwise_submission_ensemble(pseudolabels_path, model_weights, output_dir, n_folds=5):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    model_names = list(model_weights['cohesion'].keys())
    model_paths = [os.path.join(pseudolabels_path, f'{model_id}_pseudolabels') for model_id in model_names]

    df2 = pd.read_csv(os.path.join(pseudolabels_path, 'model2_pseudolabels', 'pseudolabels_fold0.csv'))
    df2 = df2.reset_index()

    target_columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    for fold in range(n_folds):
        ensemble = pd.read_csv(os.path.join(pseudolabels_path, 'model2_pseudolabels', 'pseudolabels_fold0.csv'))
        ensemble[target_columns] = 0

        for model_path, model_name in zip(model_paths, model_names):
            df = pd.read_csv(os.path.join(model_path, f'pseudolabels_fold{fold}.csv'))

            if 'in_fb3' in df.columns:
                df.drop('in_fb3', axis=1, inplace=True)

            df = pd.merge(df, df2[['text_id', 'in_fb3']], on='text_id', how='left')
            df['in_fb3'] = df['in_fb3'].astype(bool)
            df = df[~df['in_fb3']]

            if not (df.text_id.values[0] == '73D6F19E24BD' and df.text_id.values[-1] == '6D2AD3027292'):
                print(model_path, fold, ' not in right order, fixing...')

                df = pd.merge(df, df2[['text_id', 'index']], on=['text_id'], how='left')
                df = df.sort_values('index')
                df = df.reset_index(drop=True)
                df.drop(['index'], axis=1, inplace=True)

                if not (df.text_id.values[0] == '73D6F19E24BD' and df.text_id.values[-1] == '6D2AD3027292'):
                    print(model_path, fold, ' not fixed')

            if 'full_text' in df.columns:
                df.drop('full_text', axis=1, inplace=True)
            df.columns = [col + '_' + model_name if col != 'text_id' else col for col in df.columns]
            ensemble = pd.merge(ensemble, df, on='text_id', how='left')

        for col in target_columns:
            w_ = model_weights[col]
            for fn, w in w_.items():
                ensemble[col] += ensemble[col + '_' + fn] * w

        ensemble = ensemble[['text_id'] + target_columns]
        ensemble[target_columns] = ensemble[target_columns].clip(1, 5)

        ensemble = pd.merge(ensemble, df2[['text_id', 'full_text']], on=['text_id'], how='left')

        ensemble.to_csv(os.path.join(output_dir, f'pseudolabels_fold{fold}.csv'), index=False)
    return True


def make_columnwise_submission_ensemble2(filepaths, pseudolabels_path, model_weights, output_dir, n_folds=5):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    model_names = list(model_weights['cohesion'].keys())
    model_paths = [os.path.join(pseudolabels_path, f'{model_id}_pseudolabels') for model_id in model_names]

    df2 = pd.read_csv(filepaths['TRAIN_CSV_PATH'])
    df2 = df2.reset_index()

    target_columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    for fold in range(n_folds):
        ensemble = pd.read_csv(filepaths['TRAIN_CSV_PATH'])
        ensemble[target_columns] = 0

        for model_path, model_name in zip(model_paths, model_names):
            df = pd.read_csv(os.path.join(model_path, 'pseudolabels_fold0.csv'))

            df = pd.merge(df, df2[['text_id', 'index']], on=['text_id'], how='left')
            df = df.sort_values('index')
            df = df.reset_index(drop=True)
            df.drop(['index'], axis=1, inplace=True)

            if 'full_text' in df.columns:
                df.drop('full_text', axis=1, inplace=True)

            df.columns = [col + '_' + model_name if col != 'text_id' else col for col in df.columns]
            ensemble = pd.merge(ensemble, df, on='text_id', how='left')

        for col in target_columns:
            w_ = model_weights[col]
            for fn, w in w_.items():
                ensemble[col] += ensemble[col + '_' + fn] * w

        ensemble = ensemble[['text_id'] + target_columns]
        ensemble[target_columns] = ensemble[target_columns].clip(1, 5)

        ensemble = pd.merge(ensemble, df2[['text_id', 'full_text']], on=['text_id'], how='left')

        ensemble.to_csv(os.path.join(output_dir, f'pseudolabels_fold{fold}.csv'), index=False)
    return True
