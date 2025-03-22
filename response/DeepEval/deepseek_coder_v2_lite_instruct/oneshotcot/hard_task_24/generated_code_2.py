import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add convolutional layer (1x1)
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Define branches
    def branch(input_tensor, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        return x
    
    # Step 4: Create branches
    branch1 = branch(conv1x1, filters=32)
    branch2 = branch(conv1x1, filters=32)
    branch3 = branch(conv1x1, filters=32)
    
    # Step 4.5: Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Step 5: Add another 1x1 convolutional layer
    conv1x1_after_concat = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Step 6: Add batch normalization
    batch_norm = BatchNormalization()(conv1x1_after_concat)
    
    # Step 7: Add flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Step 8: Add dense layer (fully connected layer)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    
    # Step 9: Add dense layer (fully connected layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Step 10: Add final dense layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 11: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model