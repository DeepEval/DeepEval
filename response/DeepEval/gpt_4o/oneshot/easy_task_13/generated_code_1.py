import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 Convolutional Layer with Dropout
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.2)(conv1)
    
    # Second 1x1 Convolutional Layer with Dropout
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.2)(conv2)
    
    # 3x1 Convolutional Layer with Dropout
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout2)
    dropout3 = Dropout(rate=0.2)(conv3)
    
    # 1x3 Convolutional Layer with Dropout
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(dropout3)
    dropout4 = Dropout(rate=0.2)(conv4)
    
    # Restoring channel count with 1x1 Convolutional Layer
    restore_channels = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout4)
    dropout5 = Dropout(rate=0.2)(restore_channels)
    
    # Adding the processed features to the original input
    added = Add()([input_layer, dropout5])
    
    # Flattening and Fully Connected Layer
    flatten_layer = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model