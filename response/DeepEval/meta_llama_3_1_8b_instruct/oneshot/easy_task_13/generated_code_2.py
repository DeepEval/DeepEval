import keras
from keras.layers import Input, Conv2D, Add, Dropout, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # Two 1x1 convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_dropout = Dropout(0.2)(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_dropout)
    conv2_dropout = Dropout(0.2)(conv2)

    # 3x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2_dropout)
    conv3_dropout = Dropout(0.2)(conv3)

    # 1x3 convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3_dropout)
    conv4_dropout = Dropout(0.2)(conv4)

    # Restore channels to match input's channel count using another 1x1 convolutional layer
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4_dropout)
    
    # Combine the processed features with the original input via addition
    combined = Add()([input_layer, conv5])
    
    # Flattening layer
    flatten_layer = Flatten()(combined)
    
    # Fully connected layer to produce the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model