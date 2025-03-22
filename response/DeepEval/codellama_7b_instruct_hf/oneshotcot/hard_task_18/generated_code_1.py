import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    output1 = Concatenate()([conv1, conv2, avg_pool])

    # Second block
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output1)
    global_avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path)
    refined_weights = Dense(units=64, activation='relu')(global_avg_pool)
    refined_weights = Dense(units=64, activation='relu')(refined_weights)
    final_output = Flatten()(refined_weights)
    final_output = Dense(units=10, activation='softmax')(final_output)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=final_output)
    return model