import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
    
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv3)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2)(pool1)
    
    # Concatenate layers
    concat = Concatenate()([pool2, conv2, conv1, pool1])  # Adjust the order if needed
    
    # Flatten and fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model