import keras
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
    add = Add()([conv3, max_pooling])
    flatten = Flatten()(add)

    # Branch path
    branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch)

    # Combine main and branch path
    combined = Add()([flatten, branch_pooling])

    # Fully connected layer
    output = Dense(units=10, activation='softmax')(combined)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output)
    return model