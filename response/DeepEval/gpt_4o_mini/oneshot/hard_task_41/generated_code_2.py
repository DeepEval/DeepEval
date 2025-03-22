import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Dense, Reshape, Concatenate, Conv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    flattened_path1 = Flatten()(path1)
    flattened_path2 = Flatten()(path2)
    flattened_path3 = Flatten()(path3)

    dropout_path1 = Dropout(0.5)(flattened_path1)
    dropout_path2 = Dropout(0.5)(flattened_path2)
    dropout_path3 = Dropout(0.5)(flattened_path3)

    concat_block1 = Concatenate()([dropout_path1, dropout_path2, dropout_path3])
    
    # Fully connected layer after Block 1
    dense_block1 = Dense(units=256, activation='relu')(concat_block1)
    reshaped_output = Reshape((16, 16, 1))(dense_block1)  # Reshape for Block 2

    # Block 2
    path1_b2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(reshaped_output)
    path2_b2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(reshaped_output)
    path3_b2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(reshaped_output)
    path4_b2 = Concatenate()([Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(reshaped_output),
                               Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3_b2)])

    path5_b2 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshaped_output)
    path6_b2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path5_b2)

    # Concatenate outputs from Block 2
    concat_block2 = Concatenate()([path1_b2, path2_b2, path3_b2, path4_b2, path6_b2])

    # Flatten and Fully Connected Layers
    flatten_block2 = Flatten()(concat_block2)
    dense_block2 = Dense(units=128, activation='relu')(flatten_block2)
    dense_block3 = Dense(units=64, activation='relu')(dense_block2)
    
    output_layer = Dense(units=10, activation='softmax')(dense_block3)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model