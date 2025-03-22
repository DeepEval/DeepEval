import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Lambda, BatchNormalization, Flatten, Dense
from tensorflow import split

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Lambda(lambda x: split(x, num_or_size_splits=3, axis=2))(input_layer)
    
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    conv3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    x = Concatenate()([conv1, conv2, conv3])
    dropout1 = Dropout(rate=0.2)(x)

    # Block 2
    x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    x2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    x2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x2)
    x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropout1)
    x3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x3)
    
    x = Concatenate()([x1, x2, x3])

    # Flatten and Dense layers
    x = Flatten()(x)
    dense1 = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model