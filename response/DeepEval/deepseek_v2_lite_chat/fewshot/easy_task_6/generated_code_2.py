import keras
from keras.layers import Input, Conv2D, Add, BatchNormalization, Activation, concatenate, Flatten, Dense

def dl_model():
    # Path 1: Main path
    input_path1 = Input(shape=(28, 28, 1))
    conv1_main = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_path1)
    bn1_main = BatchNormalization()(conv1_main)
    relu1_main = Activation('relu')(bn1_main)
    
    conv2_main = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(relu1_main)
    bn2_main = BatchNormalization()(conv2_main)
    relu2_main = Activation('relu')(bn2_main)
    
    # Path 2: Branch path
    input_path2 = Input(shape=(28, 28, 1))
    conv3_branch = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_path2)
    bn3_branch = BatchNormalization()(conv3_branch)
    relu3_branch = Activation('relu')(bn3_branch)
    
    # Add paths together
    concat = concatenate([relu2_main, relu3_branch])
    
    # Final convolution layer
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(concat)
    bn4 = BatchNormalization()(conv4)
    relu4 = Activation('relu')(bn4)
    
    # Flatten and fully connected layers
    flat = Flatten()(relu4)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=[input_path1, input_path2], outputs=output)
    
    return model