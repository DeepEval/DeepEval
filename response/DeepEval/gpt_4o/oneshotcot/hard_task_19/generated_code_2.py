import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Branch Path
    global_avg_pooling = GlobalAveragePooling2D()(conv3)
    dense1_branch = Dense(units=32, activation='relu')(global_avg_pooling)
    dense2_branch = Dense(units=64, activation='sigmoid')(dense1_branch)
    channel_weights = Reshape((1, 1, 64))(dense2_branch)
    
    # Multiply branch output with main path
    scaled_main_path = Multiply()([conv3, channel_weights])

    # Add scaled main path and the output of main path max pooling
    combined = Add()([max_pooling, scaled_main_path])

    # Final Dense Layers for Classification
    flatten = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model