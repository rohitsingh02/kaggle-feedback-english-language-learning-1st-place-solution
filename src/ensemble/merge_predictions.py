from .get_oofs_filenames import get_oofs_filenames
import os
import pandas as pd
import numpy as np


def get_merged_predictions(train_csv_path='../data/raw/train.csv',
                           oofs_dir_path='../data/oofs/csv'):
    train_df = pd.read_csv(train_csv_path)

    oofs_filenames = get_oofs_filenames(oofs_dir_path)
    for fn in oofs_filenames:
        path = os.path.join(oofs_dir_path, f'{fn}.csv')
        oof_df = pd.read_csv(path)

        if fn.startswith('exp'):
            oof_df.drop(['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions'],
                        axis=1,
                        inplace=True)

        oof_df.rename(columns={'cohesion': 'pred_cohesion',
                               'syntax': 'pred_syntax',
                               'vocabulary': 'pred_vocabulary',
                               'phraseology': 'pred_phraseology',
                               'grammar': 'pred_grammar',
                               'conventions': 'pred_conventions'}, inplace=True)

        oof_df = oof_df[['text_id', 'pred_cohesion', 'pred_syntax',
                         'pred_vocabulary', 'pred_phraseology', 'pred_grammar', 'pred_conventions']]

        oof_df.columns = [col + '_' + fn if col != 'text_id' else col for col in oof_df.columns]
        train_df = pd.merge(train_df, oof_df, on=['text_id'], how='left')
    return train_df


def get_oofs_scores(dataframe, oofs_filenames, target_columns, criterion):
    submissions_cv_scores = {}
    for fn in oofs_filenames:
        pred_columns = ['pred_' + col + '_' + fn for col in target_columns]
        score, scores_all = criterion(dataframe[target_columns].values, dataframe[pred_columns].values)

        scores = {}
        for col, col_score in zip(target_columns, scores_all):
            scores[col] = col_score

        submissions_cv_scores[fn] = scores

    return submissions_cv_scores


def print_model_score(model, submission_cv_scores):
    scores = submission_cv_scores[model]
    print(f'================= {model} =================')
    print(f'CV score: {np.mean(list(scores.values()))}')
    for target, rmse in scores.items():
        print(f'\t{target}: {rmse}')


def print_scores(submission_cv_scores):
    for model in submission_cv_scores.keys():
        print_model_score(model, submission_cv_scores)
