import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Add, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    # Split the input into three groups
    split_layers = []
    for i in range(3):
        if i == 0:
            kernel_size = (1, 1)
        elif i == 1:
            kernel_size = (3, 3)
        else:
            kernel_size = (5, 5)
        split_layers.append(Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3)[i])(inputs))
    
    # Process each split through a separable convolutional layer and batch normalization
    conv_layers = []
    for split in split_layers:
        x = Conv2D(32, kernel_size, padding='same', activation='relu', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(split)
        x = BatchNormalization()(x)
        x = Conv2D(32, (1, 1), activation='relu')(x)
        x = BatchNormalization()(x)
        conv_layers.append(x)
    
    # Concatenate the outputs of the three groups
    x = Concatenate(axis=3)(conv_layers)

    # Block 2
    # Path 1: 1x1 convolution
    path1 = Conv2D(64, (1, 1), activation='relu')(x)

    # Path 2: 3x3 average pooling followed by 1x1 convolution
    path2 = Conv2D(64, (1, 1), activation='relu')(AveragePooling2D(pool_size=(3, 3))(x))

    # Path 3: 1x1 convolution followed by two sub-paths
    path3_input = Conv2D(64, (1, 1), activation='relu')(x)
    path3_1 = Conv2D(64, (1, 3), activation='relu')(path3_input)
    path3_2 = Conv2D(64, (3, 1), activation='relu')(path3_input)
    path3 = Concatenate(axis=3)([path3_1, path3_2])

    # Path 4: 1x1 convolution followed by 3x3 convolution, then two sub-paths
    path4_input = Conv2D(64, (1, 1), activation='relu')(x)
    path4_1 = Conv2D(64, (1, 3), activation='relu')(path4_input)
    path4_2 = Conv2D(64, (3, 1), activation='relu')(path4_input)
    path4 = Concatenate(axis=3)([path4_1, path4_2])

    # Concatenate the outputs of the four paths
    x = Concatenate(axis=3)([path1, path2, path3, path4])

    # Flatten and fully connected layer
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model