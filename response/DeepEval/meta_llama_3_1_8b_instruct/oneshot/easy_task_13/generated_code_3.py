import keras
from keras.layers import Input, Conv2D, Add, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv4 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
    
    # Combine the output of conv2 and conv5 to match the input's channel count
    combined_layer = Add()([conv2, conv5])
    
    # Add dropout layers to mitigate overfitting
    dropout1 = Dropout(0.2)(conv3)
    dropout2 = Dropout(0.2)(conv4)
    dropout3 = Dropout(0.2)(conv5)
    
    # Combine the output of the convolutional layers
    combined_layer = Add()([dropout1, dropout2, dropout3, combined_layer])
    
    # Add a dropout layer after combination
    dropout4 = Dropout(0.2)(combined_layer)
    
    # Flatten the output
    flatten_layer = Flatten()(dropout4)
    
    # Add a fully connected layer to produce the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model