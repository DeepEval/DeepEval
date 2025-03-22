import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer with 3 channels
    input_layer = Input(shape=(32, 32, 3))

    # Increase the dimensionality of the input channels with a 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract initial features using a 3x3 depthwise separable convolution
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False, depthwise_initializer='he_normal')(conv1)

    # Compute channel attention weights through global average pooling
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    attention_weights = BatchNormalization()(avg_pool)

    # Generate weights whose size is same as the channels of the initial features
    attention_weights = Flatten()(attention_weights)
    attention_weights = Dense(units=32, activation='relu')(attention_weights)

    # Reshape the attention weights to match the initial features
    attention_weights = Reshape(target_shape=(1, 1, 32))(attention_weights)

    # Multiply the attention weights with the initial features to achieve channel attention weighting
    attention_weights = Multiply()([attention_weights, conv2])

    # A 1x1 convolution reduces the dimensionality
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(attention_weights)

    # Combine the output with the initial input
    output_layer = Concatenate()([input_layer, conv3])

    # Flatten the output and pass through a fully connected layer to complete the classification process
    output_layer = Flatten()(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model