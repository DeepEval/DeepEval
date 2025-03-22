import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have a shape of 32x32 with 3 color channels
    
    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    
    # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(branch3)

    # Concatenating the outputs of all three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 Convolution to adjust output dimensions
    conv_output = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Adding the main path (input) to the processed output
    added_output = Add()([input_layer, conv_output])

    # Flatten the result
    flatten_layer = Flatten()(added_output)
    
    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# You can create the model by calling the function
model = dl_model()
model.summary()  # To check the model architecture