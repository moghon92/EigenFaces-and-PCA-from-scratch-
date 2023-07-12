import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
import sklearn.preprocessing as pp

# get test images
def get_test_images(scale=True):
    path = os.getcwd() + '\\data\\yalefaces'
    images = []
    for file in os.listdir(path):
        if file in ['subject01-test.gif', 'subject02-test.gif']:
            image_path = os.path.join(path, file)
            image = imageio.v2.imread(image_path)
            image = block_reduce(image, block_size=(4, 4), func=np.mean)
            image_flat = image.flatten()
            images.append(image_flat)

    m, n = image.shape
    images = np.vstack(images)

    if scale:
        images = pp.scale(images, axis=1)

    return images, (m, n)

# get non test images
def get_images(subject, scale=True):
    path = os.getcwd() + '\\data\\yalefaces'
    images = []
    for file in os.listdir(path):
        if file.startswith(subject) and file != f'{subject}-test.gif':
            image_path = os.path.join(path, file)
            image = imageio.v2.imread(image_path)
            image = block_reduce(image, block_size=(4, 4), func=np.mean)
            image_flat = image.flatten()
            images.append(image_flat)

    m, n = image.shape
    images = np.vstack(images)

    if scale:
        images = pp.scale(images, axis=1)
    return images, (m, n)

def calc_eigen_faces(images, k=2):
    '''
    :param images: takes a matrix of images
    :return: eigen faces
    '''
    pca = PCA(n_components=k)
    eigen_faces = pca.fit(images).components_

    return eigen_faces


def main():
    ## Q5a
    ## calc eigen faces
    subject = "subject01"
    images, imshape = get_images(subject, scale=True)
    eigen_faces = calc_eigen_faces(images, k=6)

    # plot them
    for i in range(eigen_faces.shape[0]):
        plt.imshow(eigen_faces[i].reshape(imshape))
        plt.title(f'{subject} PC0{i}')
        plt.savefig(f'{subject} EigenFace_{i}.png')

    ##Q5b

    #get images
    images_1, _ = get_images("subject01", scale=True)
    images_2, _ = get_images("subject02", scale=True)
    test_images, _ = get_test_images(scale=True)
    test_image1 = test_images[0]
    test_image2 = test_images[1]

    # calc eigen faces
    eigen_faces_S1 = calc_eigen_faces(images_1, k=6)
    eigen_faces_S2 = calc_eigen_faces(images_2, k=6)

    # calc residuals
    s11 = np.linalg.norm( test_image1 - (eigen_faces_S1.T@eigen_faces_S1@test_image1) )**2
    s12 = np.linalg.norm( test_image1 - (eigen_faces_S2.T@eigen_faces_S2@test_image1) )**2
    s21 = np.linalg.norm( test_image2 - (eigen_faces_S1.T@eigen_faces_S1@test_image2) )**2
    s22 = np.linalg.norm( test_image2 - (eigen_faces_S2.T@eigen_faces_S2@test_image2) )**2

    print('S11:- ', s11)
    print('S12:- ', s12)
    print('S21:- ', s21)
    print('S22:- ', s22)

if __name__ == "__main__":
    main()