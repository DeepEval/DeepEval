import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # First Block: Main Path & Branch Path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)  
    branch_path = input_layer 
    adding_layer = Add()([conv2, branch_path])

    # Second Block: Multiple Scale Pooling
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(adding_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(adding_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(adding_layer)

    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    concat_output = Concatenate()([flatten1, flatten2, flatten3])

    # Fully Connected Layers
    dense1 = Dense(units=64, activation='relu')(concat_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model