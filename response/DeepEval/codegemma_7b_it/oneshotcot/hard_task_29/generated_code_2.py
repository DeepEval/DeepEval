import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Add, Flatten, Concatenate
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    block1_output = Add()([main_path, branch_path])

    # Block 2
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(block1_output)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_output)
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(block1_output)

    block2_output = Concatenate()([max_pooling1, max_pooling2, max_pooling3])

    # Fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model