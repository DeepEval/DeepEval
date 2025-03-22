import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Flatten, Add, Reshape
from keras.models import Model

def dl_model():
    # Step 1: Input Layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Step 2: First Block
    # Step 2.1: First Convolutional Layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    # Step 2.2: Second Convolutional Layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    # Step 2.3: Average Pooling Layer
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Adding the input to the output of the first block via addition
    block1_output = Add()([input_layer, avg_pool1])

    # Step 3: Second Block
    # Step 3.1: Global Average Pooling Layer to generate channel weights
    gap = GlobalAveragePooling2D()(block1_output)
    
    # Step 3.2: Two Fully Connected Layers
    dense1 = Dense(units=32, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Step 3.3: Reshape for multiplication
    reshaped_weights = Reshape((1, 1, 32))(dense2)
    
    # Step 3.4: Multiply the input of the second block by the weights
    multiplied_output = Add()([block1_output, reshaped_weights])

    # Step 4: Flatten and Final Fully Connected Layer
    flatten_layer = Flatten()(multiplied_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Building the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model