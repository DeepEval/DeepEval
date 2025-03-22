import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPooling2D, Add, Flatten
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are of shape 32x32x3

    # Block 1
    global_avg_pool = GlobalAveragePooling2D()(input_layer)  # Global Average Pooling
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)  # Fully Connected Layer
    dense2 = Dense(units=32, activation='relu')(dense1)  # Fully Connected Layer
    reshaped_weights = Reshape((1, 1, 32))(dense2)  # Reshape to match input dimensions
    weighted_output = Multiply()([input_layer, reshaped_weights])  # Multiply input with weights

    # Block 2
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)  # First Convolutional Layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)  # Second Convolutional Layer
    max_pooling = MaxPooling2D(pool_size=(2, 2))(conv2)  # Max Pooling Layer

    # Combine outputs from Block 1 and Block 2
    combined_output = Add()([weighted_output, max_pooling])  # Addition of both paths

    # Final classification layers
    flatten_layer = Flatten()(combined_output)  # Flatten the output
    final_dense1 = Dense(units=128, activation='relu')(flatten_layer)  # Fully Connected Layer
    output_layer = Dense(units=10, activation='softmax')(final_dense1)  # Output Layer for 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)  # Construct the model

    return model