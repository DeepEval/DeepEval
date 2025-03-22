import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    # Step 1: Define the input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Step 2: Increase dimensionality of the input's channels threefold with a 1x1 convolution
    initial_features = Conv2D(filters=3*32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Extract initial features using a 3x3 depthwise separable convolution
    depthwise_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', 
                            depthwise_multiplier=1)(initial_features)

    # Step 4: Compute channel attention weights
    global_avg_pool = GlobalAveragePooling2D()(depthwise_conv)
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3*32, activation='sigmoid')(dense1)  # Output size matches the number of channels in initial features

    # Step 5: Reshape the weights to match the initial features
    attention_weights = Reshape((1, 1, 3*32))(dense2)

    # Step 6: Apply the attention weights to the initial features
    attention_output = Multiply()([initial_features, attention_weights])

    # Step 7: Reduce the dimensionality with a 1x1 convolution
    reduced_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(attention_output)

    # Step 8: Combine output with the initial input
    combined_output = keras.layers.add([initial_features, reduced_output])

    # Step 9: Flatten the result
    flatten_layer = Flatten()(combined_output)

    # Step 10: Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model