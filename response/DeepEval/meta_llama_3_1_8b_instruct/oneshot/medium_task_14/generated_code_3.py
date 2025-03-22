import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch_norm1)
    
    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(relu1)
    batch_norm2 = BatchNormalization()(conv2)
    relu2 = ReLU()(batch_norm2)
    
    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(relu2)
    batch_norm3 = BatchNormalization()(conv3)
    relu3 = ReLU()(batch_norm3)
    
    # Parallel branch
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None)(input_layer)
    batch_norm4 = BatchNormalization()(conv4)
    relu4 = ReLU()(batch_norm4)
    
    # Path outputs
    path1_output = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None)(relu3)
    path2_output = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None)(relu4)
    
    # Add outputs from all paths
    added_output = Add()([path1_output, path2_output, relu3])
    
    # Flatten the output
    flatten_layer = Flatten()(added_output)
    
    # Classification layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model