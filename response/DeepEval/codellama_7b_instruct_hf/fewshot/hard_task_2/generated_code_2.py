import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the convolutional layers for each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])

    # Define the max pooling layer for the main path
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Define the addition layer to combine the main path with the original input
    adding_layer = Add()([main_path, input_layer])

    # Flatten the output of the addition layer
    flatten_layer = Flatten()(adding_layer)

    # Define the fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model