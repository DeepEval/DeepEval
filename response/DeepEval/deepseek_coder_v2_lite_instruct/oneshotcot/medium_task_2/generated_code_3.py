import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # Step 2: Add first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Step 2: Add second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    # Step 3: Add max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    # Branch path
    # Step 4.1: Add convolutional layer as the first path
    branch_conv = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine both paths
    # Step 4.5: Add concatenate layer to merge the above paths
    combined = Concatenate()([max_pooling, branch_conv])
    
    # Step 5: Add batch normalization layer
    batch_norm = BatchNormalization()(combined)
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Step 7 & 8: Add dense layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 9: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model