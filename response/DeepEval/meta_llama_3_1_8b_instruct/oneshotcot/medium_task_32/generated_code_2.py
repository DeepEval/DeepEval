import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras import backend as K

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: K.concatenate([x[:,:,:,i] for i in range(3)]))(input_layer)
    
    # Define the depthwise separable convolutional layer block
    def block(input_tensor):
        path1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=6, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
        
    # Apply the block to each group
    group1 = block(split_layer[:, :, :, 0:32])
    group2 = block(split_layer[:, :, :, 32:64])
    group3 = block(split_layer[:, :, :, 64:96])
    
    # Concatenate the outputs of the three groups
    output_tensor = Concatenate()([group1, group2, group3])
    
    # Apply batch normalization
    batch_norm = BatchNormalization()(output_tensor)
    
    # Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Apply a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model