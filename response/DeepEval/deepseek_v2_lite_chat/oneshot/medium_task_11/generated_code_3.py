import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPool2D, Concatenate, Activation, AveragePooling2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Global average pooling path
    avg_pool = GlobalAveragePooling2D()(conv1)
    dense1 = Dense(units=512, activation='relu')(avg_pool)
    dense2 = Dense(units=256, activation='relu')(dense1)

    # Global max pooling path
    max_pool = GlobalMaxPool2D()(conv1)
    dense3 = Dense(units=512, activation='relu')(max_pool)
    dense4 = Dense(units=256, activation='relu')(dense3)

    # Concatenate the outputs from both paths
    concat = Concatenate()([dense1, dense3])

    # Activation to generate channel attention weights
    channel_attention_weights = Activation('sigmoid')(concat)

    # Element-wise multiplication with the original features
    fused_features = conv1 * channel_attention_weights

    # Average and max pooling operations to extract spatial features
    avg_pooling = AveragePooling2D(pool_size=(4, 4))(fused_features)
    max_pooling = MaxPooling2D(pool_size=(2, 2))(fused_features)

    # Concatenate along the channel dimension
    concatenated = Concatenate()([avg_pooling, max_pooling])

    # Flatten and feed into fully connected layers
    flatten = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])