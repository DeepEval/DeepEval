import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    conv1_3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)
    main_path = Concatenate()([conv1_1, conv1_2, conv1_3])
    
    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Merge paths
    adding_layer = Add()([main_path, branch_path])
    
    # Classification layers
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model