import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Multiply, Concatenate, AveragePooling2D
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 2: Add convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Step 3: Add batch normalization layer
    batch_norm = BatchNormalization()(conv)

    # Step 4: Add ReLU activation
    relu = ReLU()(batch_norm)

    # Step 5: Add global average pooling
    global_avg_pooling = GlobalAveragePooling2D()(relu)

    # Step 6: Fully connected layers
    dense1 = Dense(units=32, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Step 7: Reshape the output to match the initial feature map size
    reshape_layer = keras.layers.Reshape((1, 1, 32))(dense2)

    # Step 8: Multiply the reshaped output with the initial features to generate weighted feature maps
    weighted_features = Multiply()([reshape_layer, relu])

    # Step 9: Concatenate the weighted features with the input layer
    concatenated = Concatenate()([input_layer, weighted_features])

    # Step 10: Reduce dimensionality using a 1x1 convolution
    conv_final = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Step 11: Downsample using average pooling
    pooled_output = AveragePooling2D(pool_size=(2, 2), strides=2)(conv_final)

    # Step 12: Fully connected layer for classification output
    flatten_layer = GlobalAveragePooling2D()(pooled_output)  # Flattening for the Dense layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model