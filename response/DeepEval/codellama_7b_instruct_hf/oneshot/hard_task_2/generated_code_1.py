import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    # Split the input into three groups along the channel dimension
    groups = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    # Sequentially pass each group through a series of convolutions
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[2])
    # Combine the outputs from the three groups using an addition operation
    main_path = Concatenate()([conv1, conv2, conv3])
    # Fuse the main path with the original input layer through another addition
    combined_features = main_path + input_layer
    # Flatten the combined features into a one-dimensional vector
    flatten_layer = Flatten()(combined_features)
    # Feed the flattened vector into a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model