import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    # Conv2D (3x3) to extract spatial features
    main_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # 1x1 Conv2D for inter-channel information
    main_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_conv)
    # MaxPooling2D
    main_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_conv)
    # Dropout layer to mitigate overfitting
    main_pooling = BatchNormalization()(main_pooling)
    main_pooling = keras.layers.Dropout(0.5)(main_pooling)
    
    # Branch pathway
    branch_input = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_pooling)
    
    # Global average pooling to reduce dimensionality
    main_pooling = AveragePooling2D(pool_size=(3, 3))(main_pooling)
    main_pooling = Flatten()(main_pooling)
    
    # Concatenate outputs from both pathways
    concatenated = Concatenate()([main_pooling, branch_input])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model