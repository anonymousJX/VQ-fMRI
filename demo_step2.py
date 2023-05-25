from dataset_handle import get_fMRI_digits_dataset
from fMRI_embedding import fMRI_decoding_train

train_fMRI_loader, test_fMRI_loader = get_fMRI_digits_dataset()
fMRI_decoding_train(3092, 28, 16, 8, 4, 1, train_fMRI_loader, test_fMRI_loader, 100)