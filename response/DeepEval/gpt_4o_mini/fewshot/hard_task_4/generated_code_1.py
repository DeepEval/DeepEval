import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Increase the channel dimensionality with 1x1 convolution
    initial_features = Conv2D(filters=3 * 32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Extract features with a 3x3 depthwise separable convolution
    feature_extracted = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_features)

    # Step 3: Global average pooling to compute channel attention weights
    gap = GlobalAveragePooling2D()(feature_extracted)

    # Step 4: Fully connected layers to generate channel attention weights
    fc1 = Dense(units=32, activation='relu')(gap)
    fc2 = Dense(units=3 * 32, activation='sigmoid')(fc1)

    # Step 5: Reshape the weights to match the initial features
    reshaped_weights = Reshape((1, 1, 3 * 32))(fc2)

    # Step 6: Apply channel attention weighting
    channel_attended = Multiply()([initial_features, reshaped_weights])

    # Step 7: Reduce dimensionality with a 1x1 convolution
    reduced_features = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_attended)

    # Step 8: Combine the reduced features with the initial input
    combined_output = Add()([reduced_features, initial_features])

    # Step 9: Flatten the output and pass through a fully connected layer for classification
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model