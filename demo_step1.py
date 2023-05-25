from dataset_handle import get_mnist_dataset
from img_embedding import image_embedding_train

train_mnist_loader, test_mnist_loader = get_mnist_dataset()
image_embedding_train(28, 16, 8, 4, 1, train_mnist_loader, test_mnist_loader, 60)
