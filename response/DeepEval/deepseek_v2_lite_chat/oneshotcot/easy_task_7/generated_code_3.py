import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    drop1 = Dropout(0.2)(conv1)  # First dropout block
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    drop2 = Dropout(0.2)(conv2)  # Second dropout block

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    drop3 = Dropout(0.2)(conv3)  # Third dropout block

    # Restore number of channels
    conv4 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='linear')(conv3)

    # Branch path
    branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='linear')(input_layer)

    # Combine outputs from both paths
    combined = Concatenate()([conv4, branch])

    # Batch normalization and flattening
    bn = BatchNormalization()(combined)
    flat = Flatten()(bn)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model