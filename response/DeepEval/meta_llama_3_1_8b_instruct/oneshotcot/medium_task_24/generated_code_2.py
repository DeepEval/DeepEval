import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch 2
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    
    # Branch 3
    conv7 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    
    # Concatenate outputs
    output_tensor = Concatenate()([conv2, conv6, conv7])
    
    # Add dropout to prevent overfitting
    dropout_layer = Dropout(rate=0.2)(output_tensor)
    
    # Batch normalization
    bath_norm = BatchNormalization()(dropout_layer)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout_layer2 = Dropout(rate=0.2)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout_layer2)
    dropout_layer3 = Dropout(rate=0.2)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dropout_layer3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model