import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense

def dl_model():
    # Define the input shape for the model
    input_shape = (32, 32, 3)

    # Define the first level of the residual connection structure
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_shape)
    batch1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(batch1)
    main1 = Add()([conv1, relu1])

    # Define the second level of the residual connection structure
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main1)
    batch2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(batch2)
    main2 = Add()([conv2, relu2])

    # Define the third level of the residual connection structure
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main2)
    batch3 = BatchNormalization()(conv3)
    relu3 = Activation('relu')(batch3)
    main3 = Add()([conv3, relu3])

    # Define the final layer of the model
    flatten = Flatten()(main3)
    dense = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_shape, outputs=dense)

    return model