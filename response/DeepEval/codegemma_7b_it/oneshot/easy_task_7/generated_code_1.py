import keras
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():    
    
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    drop1 = Dropout(0.25)(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(drop1)
    drop2 = Dropout(0.25)(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(drop2)
    conv4 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv3)
    
    # Branch path
    branch = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Combine outputs
    combined = Add()([conv4, branch])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model