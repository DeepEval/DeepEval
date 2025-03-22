import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    """
    This function defines a deep learning model for image classification using the CIFAR-10 dataset.

    The model consists of four branches:
    1. The first branch uses a 1x1 convolution for dimensionality reduction.
    2. The second branch extracts features by first applying a 1x1 convolution followed by a 3x3 convolution.
    3. The third branch captures larger spatial information by first using a 1x1 convolution and then a 5x5 convolution.
    4. The fourth branch first performs a 3x3 max pooling for downsampling, followed by a 1x1 convolution for further processing.

    Finally, the outputs of these branches are concatenated together, flattened, and then passed through two fully connected layers to complete a 10-class classification task.

    Returns:
        model: The constructed deep learning model.
    """

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 32x32 images with 3 color channels
    # First branch: 1x1 convolution for dimensionality reduction
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second branch: 1x1 convolution followed by a 3x3 convolution
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Third branch: 1x1 convolution followed by a 5x5 convolution
    conv4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv4)
    
    # Fourth branch: 3x3 max pooling followed by a 1x1 convolution
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(input_layer)
    conv6 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)

    # Concatenate the outputs of the four branches
    output_tensor = Concatenate()([conv1, conv3, conv5, conv6])

    # Batch normalization
    batch_norm = BatchNormalization()(output_tensor)

    # Flatten the features
    flatten_layer = Flatten()(batch_norm)

    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model