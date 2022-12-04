import os
import shutil
from utils import load_filepaths

if __name__ == '__main__':
    filepaths = load_filepaths()

    model_path = os.path.join(filepaths['MODELS_DIR_PATH'], 'model23_pretrain')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    shutil.copy(os.path.join(filepaths['MODELS_DIR_PATH'], 'model19_pretrain', 'microsoft-deberta-v3-large_fold0_best.pth'),
                os.path.join(filepaths['MODELS_DIR_PATH'], 'model23_pretrain', 'microsoft-deberta-v3-large_fold0_best.pth'))

    shutil.copy(os.path.join(filepaths['MODELS_DIR_PATH'], 'model21_pretrain', 'microsoft-deberta-v3-large_fold1_best.pth'),
                os.path.join(filepaths['MODELS_DIR_PATH'], 'model23_pretrain', 'microsoft-deberta-v3-large_fold1_best.pth'))

    shutil.copy(os.path.join(filepaths['MODELS_DIR_PATH'], 'model22_pretrain', 'microsoft-deberta-v3-large_fold2_best.pth'),
                os.path.join(filepaths['MODELS_DIR_PATH'], 'model23_pretrain', 'microsoft-deberta-v3-large_fold2_best.pth'))

    shutil.copy(os.path.join(filepaths['MODELS_DIR_PATH'], 'model21_pretrain', 'microsoft-deberta-v3-large_fold3_best.pth'),
                os.path.join(filepaths['MODELS_DIR_PATH'], 'model23_pretrain', 'microsoft-deberta-v3-large_fold3_best.pth'))

    shutil.copy(os.path.join(filepaths['MODELS_DIR_PATH'], 'model21_pretrain', 'microsoft-deberta-v3-large_fold4_best.pth'),
                os.path.join(filepaths['MODELS_DIR_PATH'], 'model23_pretrain', 'microsoft-deberta-v3-large_fold4_best.pth'))
