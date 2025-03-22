import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Multiply, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Parallel branches
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate branches
    concat = Concatenate()([branch1, branch2, branch3, branch4])

    # Second Block: Global Average Pooling and Fully Connected Layers
    global_avg_pool = GlobalAveragePooling2D()(concat)
    dense1 = Dense(units=concat.shape[-1], activation='relu')(global_avg_pool)
    dense2 = Dense(units=concat.shape[-1], activation='relu')(dense1)

    # Reshape and Multiply
    reshaped = Reshape((1, 1, concat.shape[-1]))(dense2)
    scaled_features = Multiply()([concat, reshaped])

    # Final Fully Connected Layer
    flatten = GlobalAveragePooling2D()(scaled_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model