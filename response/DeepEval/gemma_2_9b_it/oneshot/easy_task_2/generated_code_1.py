import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():     

        input_layer = Input(shape=(224, 224, 3))
        
        # First feature extraction block
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

        # Second feature extraction block
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

        # Additional convolutional layers
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
        conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)

        # Final average pooling
        pool3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv5)

        # Flatten and fully connected layers
        flat = Flatten()(pool3)
        dense1 = Dense(units=512, activation='relu')(flat)
        dropout1 = Dropout(rate=0.5)(dense1)
        dense2 = Dense(units=1000, activation='softmax')(dropout1)

        model = keras.Model(inputs=input_layer, outputs=dense2)

        return model