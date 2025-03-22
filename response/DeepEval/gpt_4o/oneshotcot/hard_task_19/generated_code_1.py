import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path: three 3x3 convolutional layers followed by a max pooling layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    main_max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv3)

    # Branch path: global average pooling and two fully connected layers
    global_avg_pooling = GlobalAveragePooling2D()(conv3)
    dense_branch_1 = Dense(units=64, activation='relu')(global_avg_pooling)
    dense_branch_2 = Dense(units=128, activation='sigmoid')(dense_branch_1)
    reshaped_weights = Reshape((1, 1, 128))(dense_branch_2)
    
    # Multiply channel weights with the input
    scaled_input = Multiply()([conv3, reshaped_weights])
    
    # Add outputs from both paths
    added_outputs = Add()([main_max_pooling, scaled_input])
    
    # Additional fully connected layers for classification
    flatten_layer = Flatten()(added_outputs)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model