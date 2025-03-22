import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # 1x1 Convolutions
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(conv1)  
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(dropout1)

    # 3x1 Convolution
    conv3 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(conv2)
    dropout3 = Dropout(0.25)(conv3)

    # 1x3 Convolution
    conv4 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu')(dropout3)
    dropout4 = Dropout(0.25)(conv4)

    # Restore channels
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(dropout4)

    # Feature Combination
    output = Add()([input_layer, conv5])

    # Flatten and Fully Connected Layer
    flatten = Flatten()(output)
    dense = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model