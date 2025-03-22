import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization, Concatenate, Flatten, Dense, Lambda

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        # main path
        path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Dropout(rate=0.5)(path1)
        path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        # branch path
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # merge outputs
        output_tensor = Concatenate()([path1, path2])
        
        return output_tensor
        
    # block 1
    block_output = block(input_tensor=input_layer)
    
    # block 2
    def block2(input_tensor):
        # split the input into three groups along the last dimension
        groups = tf.split(input_tensor, num_or_size_splits=3, axis=3)
        
        # process each group with separable convolutional layers
        path1 = Lambda(lambda x: tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(x))(groups[0])
        path1 = Dropout(rate=0.5)(path1)
        path1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path1)
        path1 = Dropout(rate=0.5)(path1)
        
        path2 = Lambda(lambda x: tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x))(groups[1])
        path2 = Dropout(rate=0.5)(path2)
        path2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path2)
        path2 = Dropout(rate=0.5)(path2)
        
        path3 = Lambda(lambda x: tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(x))(groups[2])
        path3 = Dropout(rate=0.5)(path3)
        path3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path3)
        path3 = Dropout(rate=0.5)(path3)
        
        # concatenate the outputs from the three groups
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor
        
    block_output2 = block2(input_tensor=block_output)
    
    # flatten and fully connected layer
    flatten_layer = Flatten()(block_output2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model