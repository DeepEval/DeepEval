import keras
from keras.layers import Input, DepthwiseConv2D, BatchNormalization, Conv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine Paths
    combined_path = Add()([x, branch_path])

    # Flatten and Classify
    x = Flatten()(combined_path)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model