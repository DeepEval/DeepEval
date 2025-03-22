import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Increase the dimensionality of the input's channels threefold with a 1x1 convolution
    conv1 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Extract initial features using a 3x3 depthwise separable convolution
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1)

    # Step 3: Compute channel attention weights
    gap = GlobalAveragePooling2D()(conv2)
    dense1 = Dense(units=conv2.shape[-1] // 16, activation='relu')(gap)
    dense2 = Dense(units=conv2.shape[-1], activation='relu')(dense1)
    attention_weights = Dense(units=conv2.shape[-1], activation='sigmoid')(dense2)

    # Step 4: Reshape the weights to match the initial features and multiply them with the initial features
    attention_weights = keras.backend.reshape(attention_weights, (1, 1, 1, conv2.shape[-1]))
    weighted_features = Multiply()([conv2, attention_weights])

    # Step 5: Combine the output with the initial input
    added_features = Add()([conv2, weighted_features])

    # Step 6: Flatten the result
    flattened = Flatten()(added_features)

    # Step 7: Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model