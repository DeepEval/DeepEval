import keras
from keras.layers import Input, Conv2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Path 1: Main path
    main_input = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_input)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(bn1)
    bn2 = BatchNormalization()(conv2)
    
    # Path 2: Branch path
    branch_input = Input(shape=(28, 28, 1))
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch_input)
    bn3 = BatchNormalization()(branch_conv)
    
    # Add the outputs of the main and branch paths
    add_layer = Add()([bn2, bn3])
    
    # Flatten and fully connected layer
    flatten = Flatten()(add_layer)
    dense = Dense(units=128, activation='relu')(flatten)
    output = Dense(units=10, activation='softmax')(dense)
    
    model = keras.Model(inputs=[main_input, branch_input], outputs=output)
    
    return model