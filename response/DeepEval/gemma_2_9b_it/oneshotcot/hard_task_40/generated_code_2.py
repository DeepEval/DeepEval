import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout, BatchNormalization

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # First Block: Average Pooling and Concatenation
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    concatenated = Concatenate()([flat1, flat2, flat3])
    dense1 = Dense(units=128, activation='relu')(concatenated)
    reshaped = Reshape((1, 128))(dense1)

    # Second Block: Multi-Scale Feature Extraction
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2) 
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3) 
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

        path1 = Dropout(0.2)(path1)
        path2 = Dropout(0.2)(path2)
        path3 = Dropout(0.2)(path3)
        path4 = Dropout(0.2)(path4)

        output_tensor = Concatenate(axis=3)([path1, path2, path3, path4])
        return output_tensor

    block_output = block(reshaped)

    dense2 = Dense(units=64, activation='relu')(block_output)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model