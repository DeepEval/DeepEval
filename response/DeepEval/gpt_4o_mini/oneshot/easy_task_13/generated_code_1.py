import keras
from keras.layers import Input, Conv2D, Add, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolution
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1_1 = Dropout(0.2)(conv1_1)
    
    # Second 1x1 convolution
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1_1)
    dropout1_2 = Dropout(0.2)(conv1_2)
    
    # 3x1 convolution
    conv3_1 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(dropout1_2)
    dropout3_1 = Dropout(0.2)(conv3_1)
    
    # 1x3 convolution
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(dropout3_1)
    dropout1_3 = Dropout(0.2)(conv1_3)
    
    # Final 1x1 convolution to match the original input channels
    conv_final = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(dropout1_3)
    
    # Adding the processed features with the original input
    added_output = Add()([input_layer, conv_final])
    
    # Flattening and final fully connected layer
    flatten_layer = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model