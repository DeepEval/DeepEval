import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Flatten
from keras.models import Model

def dl_model():
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Combine input with the output of the main path
    combined_output = Add()([input_layer, avg_pool1])

    # Block 2
    global_avg_pool = GlobalAveragePooling2D()(combined_output)
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)  # Generate channel weights

    # Reshape to match the dimensions of the combined output
    reshaped_weights = Reshape((1, 1, 32))(dense2)
    weighted_output = keras.layers.multiply([combined_output, reshaped_weights])  # Element-wise multiplication

    # Flatten and output layer
    flatten = Flatten()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model