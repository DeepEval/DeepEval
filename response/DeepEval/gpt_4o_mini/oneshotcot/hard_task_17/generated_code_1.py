import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    # Input Layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)

    # Block 1
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)
    
    # Fully connected layers in Block 1
    fc1 = Dense(units=32, activation='relu')(global_avg_pooling)
    fc2 = Dense(units=32, activation='relu')(fc1)
    
    # Reshape to the input shape and multiply with input layer
    reshaped_weights = Reshape((1, 1, 32))(fc2)
    weighted_output = Multiply()([input_layer, reshaped_weights])
    
    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(weighted_output)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch connecting Block 1 to Block 2
    block1_output = Multiply()([input_layer, reshaped_weights])
    
    # Combine outputs from Block 1 and Block 2 using addition
    combined_output = Add()([max_pooling, block1_output])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model