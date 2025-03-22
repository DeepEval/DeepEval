import keras
from keras.layers import Input, Conv2D, Add, LayerNormalization, Dense, BatchNormalization

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Depthwise separable convolutional layer
    dw_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same',
                      use_depthwise=True, kernel_regularizer=keras.regularizers.l2(0.01),
                      depth_kwargs={'activation': 'relu'})(input_layer)
    
    # Layer normalization
    dw_conv = LayerNormalization(axis=3)(dw_conv)
    
    # Fully connected layer for channel-wise transformation (for each channel)
    fc_1 = Dense(units=512, activation='relu')(dw_conv)
    fc_2 = Dense(units=256, activation='relu')(fc_1)
    
    # Add the processed features to the original input
    combined = Add()([input_layer, dw_conv])
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(fc_2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and return the model
model = dl_model()
model.summary()