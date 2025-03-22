import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPooling2D, Add, Flatten

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=32, activation='relu')(avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Reshape to match input shape (32, 32, 3)
    reshaped_weights = Reshape((1, 1, 32))(dense2)
    
    # Multiply weights with input to get the weighted feature output
    weighted_output = Multiply()([input_layer, reshaped_weights])
    
    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(weighted_output)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch from Block 1 connecting directly to Block 2
    block1_output = weighted_output

    # Fuse the outputs from both blocks through addition
    combined_output = Add()([max_pooling, block1_output])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers for classification
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    dense4 = Dense(units=64, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model