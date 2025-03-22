import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():  
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 Convolutional Layers
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)  

    conv2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(conv2)

    # 3x1 Convolutional Layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 1), activation='relu')(dropout2)
    dropout3 = Dropout(0.2)(conv3)

    # 1x3 Convolutional Layer
    conv4 = Conv2D(filters=128, kernel_size=(1, 3), activation='relu')(dropout3)
    dropout4 = Dropout(0.2)(conv4)

    # Restore Channels
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(dropout4)

    # Feature Combination
    output_features = Add()([input_layer, conv5])

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(output_features)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model