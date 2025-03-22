import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Apply an initial convolutional layer with 32 kernels
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Split the input into two groups along the last dimension
    split = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv)
    
    # Define the first block
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Lambda(lambda x: tf.concat([x, split[1]], axis=-1))(block1)
    
    # Define the second block
    block2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = Lambda(lambda x: tf.concat([x, block1], axis=-1))(block2)
    
    # Flatten and pass through a fully connected layer for classification
    flatten = Flatten()(block2)
    dense = Dense(units=10, activation='softmax')(flatten)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=dense)
    
    return model