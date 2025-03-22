import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3)) 

    # Main path
    conv1x1_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer) 
    
    # Second path
    conv1x7_2nd = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv7x1_2nd = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv1x7_2nd)

    # Concatenate outputs
    concat_layer = Concatenate()([conv1x1_main, conv7x1_2nd])
    
    # 1x1 convolution for alignment
    output_main = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)

    # Branch connecting directly to input
    branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add outputs from main path and branch
    output = add([output_main, branch]) 

    # Flatten and fully connected layers
    flatten_layer = Flatten()(output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model