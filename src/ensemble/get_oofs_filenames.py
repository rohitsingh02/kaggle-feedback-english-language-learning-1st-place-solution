import os


def get_oofs_filenames(dir_path='../data/oofs/csv/'):
    oofs_filenames = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv'):
            oofs_filenames.append(filename.replace('.csv', ''))
    return oofs_filenames


def filter_filenames(filenames, banned_models=None):
    if banned_models is not None:
        filtered_filenames = [fn for fn in filenames if fn not in banned_models]
    else:
        filtered_filenames = filenames
    return filtered_filenames
