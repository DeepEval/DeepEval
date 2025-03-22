from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model
from keras.regularizers import l2
from tensorflow import keras

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.

    The model consists of two consecutive basic blocks, where each block comprises a convolutional layer, batch normalization, and ReLU activation as its main path.
    The block's branch connects directly to the input, and feature fusion occurs by adding the outputs from both paths.
    An initial convolutional layer reduces the dimensionality of the input features to 16.
    The outputs from both paths are combined again through addition, enhancing feature representation.
    Finally, an average pooling layer downsamples the feature map, which is then flattened and processed through a fully connected layer to produce the output for the classification task.
    """

    # Define the input shape and number of classes
    input_shape = (32, 32, 3)
    num_classes = 10

    # Define the initial convolutional layer
    initial_conv = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=input_shape)(Input(shape=input_shape))

    # Define the basic block
    def basic_block(x):
        branch = x
        main_path = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)
        main_path = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(main_path)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)
        return Add()([branch, main_path])

    # Define the first basic block
    x = basic_block(initial_conv)

    # Define the feature extraction layer
    feature_extraction = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)

    # Define the second basic block
    x = basic_block(feature_extraction)

    # Combine the outputs from both paths
    x = Add()([x, feature_extraction])

    # Define the average pooling layer
    x = AveragePooling2D((2, 2))(x)

    # Flatten the feature map
    x = Flatten()(x)

    # Define the fully connected layer
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=Input(shape=input_shape), outputs=x)

    return model