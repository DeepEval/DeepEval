import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, multiply, Reshape

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=512, activation='relu')(x)

    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch_x = GlobalAveragePooling2D()(branch_conv)
    
    # Adjust branch_x channel to match input_layer
    branch_x = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='linear')(branch_x)
    branch_x = Reshape((3, 32, 32))(branch_x)

    # Add both paths
    combined_tensor = add([x, branch_x])
    
    # Pass through three fully connected layers
    combined_tensor = Flatten()(combined_tensor)
    combined_tensor = Dense(units=256, activation='relu')(combined_tensor)
    combined_tensor = Dense(units=128, activation='relu')(combined_tensor)
    output_layer = Dense(units=10, activation='softmax')(combined_tensor)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model