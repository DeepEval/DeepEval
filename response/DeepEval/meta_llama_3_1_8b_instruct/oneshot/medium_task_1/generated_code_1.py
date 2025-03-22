import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset images are 32x32x3
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Add the input layer to the max-pooled features
    add_layer = Add()([input_layer, max_pooling])
    
    # Since the input layer has been added to the max-pooled features, 
    # we can directly flatten the output of the add layer
    flatten_layer = Flatten()(add_layer)
    
    # Process the flattened features through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model