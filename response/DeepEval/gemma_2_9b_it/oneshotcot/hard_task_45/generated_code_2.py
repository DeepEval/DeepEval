import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, MaxPooling2D, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  
    
    # First Block: Depthwise Separable Convolutions
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    x = [
        Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch) for branch in x
    ] + [
        Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch) for branch in x
    ] + [
        Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch) for branch in x
    ]
    x = Concatenate(axis=3)(x)

    # Second Block: Multi-Branch Feature Extraction
    x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x2)
    x3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x3)
    x = Concatenate(axis=3)([x1, x2, x3])

    # Flatten and Fully Connected Layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model