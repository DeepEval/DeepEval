import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_conv1)
    main_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_pooling)
    main_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_conv2)
    
    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Concatenate main and branch paths
    concatenated = Concatenate()(
        [main_pooling_2, branch_conv]
    )
    
    # Batch normalization and flatten
    batch_norm = BatchNormalization()(concatenated)
    flatten = Flatten()(batch_norm)
    
    # Two dense layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()