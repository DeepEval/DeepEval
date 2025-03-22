import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial Convolutional Layer
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)

    # Global Average Pooling
    global_avg_pooling = GlobalAveragePooling2D()(relu)

    # Fully connected layer to match channels of initial features
    dense1 = Dense(units=64 // 2, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=64, activation='sigmoid')(dense1)  # Adjust dimensions to match channels of initial features

    # Reshape to match the size of initial feature maps
    se_reshape = Reshape((1, 1, 64))(dense2)

    # Multiply with initial features to generate weighted feature maps
    weighted_feature_maps = Multiply()([relu, se_reshape])

    # Concatenate weighted feature maps with the input layer
    concatenated = Concatenate()([input_layer, weighted_feature_maps])

    # Dimensionality reduction and downsampling using 1x1 convolution
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)

    # Average pooling
    avg_pooling = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1)

    # Fully connected layer for classification output
    output_layer = Dense(units=10, activation='softmax')(avg_pooling)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model