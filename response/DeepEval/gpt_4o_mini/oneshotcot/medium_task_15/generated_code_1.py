import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Multiply, Concatenate, AveragePooling2D, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels
    
    # Step 1: Convolutional layer
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Step 2: Batch normalization and ReLU activation
    batch_norm = BatchNormalization()(conv_layer)
    relu_activation = ReLU()(batch_norm)

    # Step 3: Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(relu_activation)
    
    # Step 4: Two fully connected layers
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Step 5: Reshape to match the number of channels of the initial features
    reshaped = Reshape((1, 1, 32))(dense2)
    
    # Step 6: Multiply the reshaped output with the initial feature maps
    weighted_features = Multiply()([relu_activation, reshaped])
    
    # Step 7: Concatenate with the input layer
    concatenated = Concatenate()([input_layer, weighted_features])
    
    # Step 8: Downsample using 1x1 convolution and average pooling
    conv_downsample = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv_downsample)
    
    # Step 9: Fully connected layer for output classification
    flatten = GlobalAveragePooling2D()(avg_pool)  # Flattening before the output layer
    output_layer = Dense(units=10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model