import keras
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(maxpool1)
    maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(maxpool2)
    maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(maxpool3)
    output_tensor = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer and reshape operation
    dense = Dense(units=64, activation='relu')(output_tensor)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)
    
    # Second block
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(reshaped)
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
    conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
    output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
    
    # Flatten and fully connected layer
    flatten = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model