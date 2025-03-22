import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    # Step 1: add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: add convolutional layer with 3x3 convolutions
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)

    # Step 3: add convolutional layer with 1x1 convolutions followed by two 3x3 convolutions
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2)

    # Step 4: add convolutional layer with max pooling
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=3, padding='same')(branch3)

    # Step 5: concatenate the outputs of the three branches
    output_tensor = Concatenate()([branch1, branch2, branch3])

    # Step 6: add batch normalization layer
    bath_norm = BatchNormalization()(output_tensor)

    # Step 7: add flatten layer
    flatten_layer = Flatten()(bath_norm)

    # Step 8: add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Step 9: add dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 10: add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model