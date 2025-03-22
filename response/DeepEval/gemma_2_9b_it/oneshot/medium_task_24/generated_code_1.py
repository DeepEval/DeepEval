import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1
    branch1_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1_conv1)
    branch1_dropout = Dropout(0.25)(branch1_conv2)

    # Branch 2
    branch2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2_conv1)
    branch2_conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2_conv2)
    branch2_conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_conv3)
    branch2_dropout = Dropout(0.25)(branch2_conv4)

    # Branch 3
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_dropout = Dropout(0.25)(branch3_conv)

    # Concatenate branches
    concatenated_output = Concatenate()([branch1_dropout, branch2_dropout, branch3_dropout])
    
    # Flatten and Dense layers
    flatten_layer = Flatten()(concatenated_output)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model