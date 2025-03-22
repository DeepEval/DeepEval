import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolutional layer with dropout
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    drop1 = Dropout(rate=0.3)(conv1)
    
    # Second 1x1 convolutional layer with dropout
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(drop1)
    drop2 = Dropout(rate=0.3)(conv2)
    
    # 3x1 convolutional layer with dropout
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(drop2)
    drop3 = Dropout(rate=0.3)(conv3)
    
    # 1x3 convolutional layer with dropout
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(drop3)
    drop4 = Dropout(rate=0.3)(conv4)
    
    # Restoring the number of channels to match the input using a 1x1 convolution
    conv_restore = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(drop4)
    drop_restore = Dropout(rate=0.3)(conv_restore)
    
    # Adding the processed features to the original input
    add_layer = Add()([drop_restore, input_layer])
    
    # Flattening and fully connected layer
    flatten_layer = Flatten()(add_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model