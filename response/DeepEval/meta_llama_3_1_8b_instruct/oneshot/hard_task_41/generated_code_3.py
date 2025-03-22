import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)

    def block1(input_tensor):
        # Average pooling layers of three different scales
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)

        # Flatten and regularization
        flat1 = Flatten()(pool1)
        flat2 = Flatten()(pool2)
        flat3 = Flatten()(pool3)
        drop1 = Dropout(0.2)(flat1)
        drop2 = Dropout(0.2)(flat2)
        drop3 = Dropout(0.2)(flat3)

        # Concatenate and fuse
        output_tensor = Concatenate()([drop1, drop2, drop3])

        return output_tensor
    
    block1_output = block1(conv)
    # Fully connected layer and reshaping operation
    fc = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape((4, 128))(fc)
    
    def block2(input_tensor):
        # Four branches for feature extraction
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(AveragePooling2D(pool_size=(3, 3), strides=3, padding='same')(conv1))

        # Concatenate and fuse
        output_tensor = Concatenate()([conv2, conv3, pool_conv])

        return output_tensor
        
    block2_output = block2(reshaped)
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model