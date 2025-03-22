import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Two parallel convolutional branches
    conv_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch1)

    conv_branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv_branch2)

    # Combine the outputs of the two branches through addition
    combined_branch = Add()([conv_branch1, conv_branch2])

    # Global average pooling layer
    global_pooling = GlobalAveragePooling2D()(combined_branch)

    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(global_pooling)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Attention weights
    attention_weights = Dense(units=10, activation='softmax')(dense2)

    # Weighted output
    weighted_output = Multiply()([combined_branch, attention_weights])
    weighted_output = GlobalAveragePooling2D()(weighted_output)

    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model