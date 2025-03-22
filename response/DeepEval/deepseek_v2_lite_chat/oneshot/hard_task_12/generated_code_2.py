import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    main_conv = Conv2D(filters=64, kernel_size=1, padding='same')(input_layer)
    main_conv = BatchNormalization()(main_conv)
    main_conv = keras.activations.relu(main_conv)
    
    main_conv2 = Conv2D(filters=64, kernel_size=3, padding='same')(input_layer)
    main_conv2 = BatchNormalization()(main_conv2)
    main_conv2 = keras.activations.relu(main_conv2)
    
    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=3, padding='same')(input_layer)
    branch_conv = BatchNormalization()(branch_conv)
    branch_conv = keras.activations.relu(branch_conv)
    
    # Concatenate outputs from main and branch paths
    concat = Concatenate(axis=-1)([main_conv, main_conv2, branch_conv])
    
    # Flatten and pass through two fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Construct the model
model = dl_model()
model.summary()