import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28,28,1))

    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    branch_path = input_layer
    
    combined = Add()([main_path, branch_path])
    
    block1_output = combined 

    # Block 2
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)

    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    block2_output = Concatenate()([flatten1, flatten2, flatten3])

    dense1 = Dense(units=128, activation='relu')(block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model