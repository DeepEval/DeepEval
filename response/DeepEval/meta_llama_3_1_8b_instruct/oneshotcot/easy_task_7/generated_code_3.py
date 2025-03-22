import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Add

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_dropout = Dropout(0.2)(block1)
    block2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_dropout)
    block2_dropout = Dropout(0.2)(block2)
    restore_channels = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_dropout)
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the main path and the branch path
    combined = Add()([restore_channels, branch_path])
    
    # Add batch normalization layer
    bath_norm = BatchNormalization()(combined)
    
    # Add flattening layer
    flatten_layer = Flatten()(bath_norm)
    
    # Add a fully connected layer to produce the final classification probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model