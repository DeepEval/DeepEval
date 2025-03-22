import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Add, Conv2D, MaxPooling2D, Activation

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of two sequential blocks:
    Block 1: Global average pooling generates weights that pass through two fully connected layers with the same channel as the input layer.
    These weights are reshaped to match the input's shape and multiplied with the input to produce the weighted feature output.
    Block 2: Extracts deep features using two 3x3 convolutional layers followed by a max pooling layer.
    A branch from Block 1 connects directly to the output of Block 2.
    The outputs from the main path and the branch are then fused through addition.
    Finally, the combined output is classified using two fully connected layers.
    """
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    avg_pooling = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=128, activation='relu')(avg_pooling)
    weights = Dense(units=128, activation='relu')(dense1)
    weights = Reshape((1, 128))(weights)
    weighted_feature_output = Multiply()([input_layer, weights])
    
    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Branch from Block 1 connects directly to the output of Block 2
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(weighted_feature_output)
    branch = Activation('relu')(branch)
    
    # Fuse the outputs from the main path and the branch
    output = Add()([main_path, branch])
    
    # Flatten the output
    flatten_layer = keras.layers.Lambda(lambda x: keras.backend.flatten(x))(output)
    
    # Classification
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model