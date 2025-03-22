import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main Pathway
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
    drop = Dropout(0.5)(pool)

    # Branch Pathway
    branch = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(input_layer)

    # Feature Fusion
    merged = Concatenate()([drop, branch])

    # Final Layers
    gap = GlobalAveragePooling2D()(merged)
    flatten = Flatten()(gap)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model