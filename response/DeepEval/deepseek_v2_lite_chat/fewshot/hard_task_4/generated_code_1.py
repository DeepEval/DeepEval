import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dense, GlobalAveragePooling2D, Lambda, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Conv layer to increase channels from 3 to 9
    conv_layer = Conv2D(9, (1, 1), padding='same')(input_layer)

    # Depthwise separable conv layer
    depthwise_conv = Conv2D(9, (3, 3), padding='same', activation='relu')(input_layer)

    # Global average pooling to compute attention weights
    avg_pool = GlobalAveragePooling2D()(depthwise_conv)
    flatten = Reshape((9, 1, 1))(avg_pool)

    # Fully connected layer for attention weights
    fc1 = Dense(1, activation='sigmoid')(flatten)
    fc2 = Dense(9, activation='sigmoid')(flatten)

    # Reshape attention weights to match the feature map dimension
    fc1 = Reshape((9,))(fc1)
    fc2 = Reshape((9, 1, 1))(fc2)

    # Element-wise multiplication of attention weights with depthwise conv features
    attention_weighted = keras.layers.multiply([depthwise_conv, fc1, fc2])

    # Reduce dimensions
    conv_attention = Conv2D(9, (1, 1), padding='same')(attention_weighted)

    # Output layer
    output_layer = Conv2D(10, (1, 1), padding='same')(conv_attention)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()