import numpy as np
import pathlib

def get_training_data():
    '''
    ---here mnist.npz file contains compressed numpy data of almost 60k handwritten digits with 28*28=784 pixel total

                                            #data pre-processing

    --- images have gray scale values, unsigned integar , but as many operations will be done we need floating point value for accuracy and 
        for using probability we ranged it from [0,1]. 0 for white and 1 for brightest meaning black.
    
    --- flattening image: Here the numpy array shape of the images are like for 67th image ->(image_67,28,28) but instead of mattrix of 28*28 
        we want it to be a flat vector of all 784 pixels for each images, so converting array shape to (image_67,784)

    --- eye() is a NumPy function that creates an identity matrix. instead of getting labels as a single value we make it a identity mattrix of 10
        this cattegorize and good for our further operations

    '''
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/mnist_data/mnist.npz") as f:
        images,labels= f["x_train"],f["y_train"]
    images=images.astype("float32")/255
    images=np.reshape(images,(images.shape[0],images.shape[1]*images.shape[2]))
    labels=np.eye(10)[labels]
    return images,labels
    