from keras.models import Model
from keras.layers import Input, Lambda, Concatenate, Dropout, Flatten, Dense
from keras.applications import VGG16


def dl_model():
    # CIFAR-10 dataset
    num_classes = 10
    input_shape = (32, 32, 3)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Split the input into three groups along the channel axis
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer)

    # Define the first block
    first_block = []
    for i in range(3):
        # Convolutional layer with a kernel size of 1x1, 3x3, or 5x5
        conv_layer = Conv2D(32, (1, 1), activation='relu')(split_layer[i])
        # Dropout layer to reduce overfitting
        dropout_layer = Dropout(0.2)(conv_layer)
        # Concatenate the outputs from all three groups
        first_block.append(dropout_layer)
    first_block = Concatenate()(first_block)

    # Define the second block
    second_block = []
    for i in range(4):
        # Convolutional layer with a kernel size of 1x1, 3x3, or 5x5
        conv_layer = Conv2D(64, (1, 1), activation='relu')(first_block)
        # Dropout layer to reduce overfitting
        dropout_layer = Dropout(0.2)(conv_layer)
        # Max pooling layer with a kernel size of 3x3
        pool_layer = MaxPooling2D(pool_size=(3, 3))(dropout_layer)
        # Convolutional layer with a kernel size of 1x1
        conv_layer = Conv2D(128, (1, 1), activation='relu')(pool_layer)
        # Concatenate the outputs from all four branches
        second_block.append(conv_layer)
    second_block = Concatenate()(second_block)

    # Flatten the output of the second block
    flatten_layer = Flatten()(second_block)

    # Fully connected layer with 10 output units
    fc_layer = Dense(num_classes, activation='softmax')(flatten_layer)

    # Define the model
    model = Model(inputs=input_layer, outputs=fc_layer)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model