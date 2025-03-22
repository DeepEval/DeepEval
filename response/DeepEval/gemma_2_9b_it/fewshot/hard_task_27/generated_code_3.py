import keras
from keras.layers import Input, Conv2D, LayerNormalization, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Spatial Feature Extraction
    depthwise_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', depth_multiplier=1, activation='relu')(input_layer)
    norm_layer = LayerNormalization()(depthwise_conv)

    # Channel-wise Feature Transformation
    flatten_layer = Flatten()(norm_layer)
    dense1 = Dense(units=32, activation='relu')(flatten_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Feature Combination
    combined_features = Add()([input_layer, dense2])

    # Classification
    flatten_combined = Flatten()(combined_features)
    dense3 = Dense(units=128, activation='relu')(flatten_combined)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model