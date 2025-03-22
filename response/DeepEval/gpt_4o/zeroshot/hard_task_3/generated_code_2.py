import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Lambda, Add, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Function to create a <1x1 convolution, 3x3 convolution> block with dropout
    def conv_block(x):
        x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.5)(x)
        return x

    # Apply the conv_block to each split
    group_1 = conv_block(split_layer[0])
    group_2 = conv_block(split_layer[1])
    group_3 = conv_block(split_layer[2])

    # Concatenate the outputs of the three groups
    main_pathway = Concatenate()([group_1, group_2, group_3])

    # Branch pathway with 1x1 convolution to match the output dimension of the main pathway
    branch_pathway = Conv2D(96, (1, 1), activation='relu', padding='same')(input_layer)

    # Combine main and branch pathways using addition
    combined = Add()([main_pathway, branch_pathway])

    # Flatten the output and add a fully connected layer for classification
    flat = Flatten()(combined)
    output_layer = Dense(10, activation='softmax')(flat)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()