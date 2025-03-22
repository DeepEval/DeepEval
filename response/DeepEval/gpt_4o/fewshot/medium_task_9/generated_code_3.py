import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, ReLU, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters, kernel_size=(3, 3), strides=(1, 1)):
    # Main path
    main_conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    main_bn = BatchNormalization()(main_conv)
    main_relu = ReLU()(main_bn)
    
    # Adding both paths (main + branch)
    output_tensor = Add()([input_tensor, main_relu])
    
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Initial convolution to reduce dimensionality
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First basic block
    block1_output = basic_block(input_tensor=initial_conv, filters=16)

    # Second basic block
    block2_output = basic_block(input_tensor=block1_output, filters=16)

    # Branch path convolution for additional feature extraction
    branch_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Feature fusion by adding outputs from both paths
    combined_output = Add()([block2_output, branch_conv])

    # Average pooling layer
    avg_pooling = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(combined_output)

    # Flatten and fully connected layer for classification
    flatten = Flatten()(avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model