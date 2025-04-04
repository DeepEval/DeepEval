import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(224, 224, 3))

    # Feature extraction layers 1 & 2
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Additional convolutional layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(avg_pool2)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv4)

    avg_pool3 = AveragePooling2D(pool_size=(2, 2))(conv5)

    # Flatten and dense layers
    flatten_layer = Flatten()(avg_pool3)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)  
    dense2 = Dense(units=1024, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)  
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model