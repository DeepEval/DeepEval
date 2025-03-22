import keras
from keras.layers import Input, Conv2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Define a convolutional block
    def conv_block(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    # Add dropout layer to the output of the convolutional block
    output_tensor = conv_block(input_layer)
    output_tensor = Dropout(0.2)(output_tensor)  # Add dropout layer
    
    # Add another 1x1 convolutional layer to restore the number of channels
    output_tensor = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)

    # Combine the output of the convolutional block and the original input
    combined_tensor = Add()([output_tensor, input_layer])

    # Add a flattening layer
    flatten_layer = Flatten()(combined_tensor)

    # Add a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model