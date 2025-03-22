import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, Add, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)

    # Define the main pathway
    main_pathway = []
    for group in split:
        # 1x1 convolution
        x = Conv2D(32, (1, 1), activation='relu')(group)
        # 3x3 convolution
        x = Conv2D(32, (3, 3), activation='relu')(x)
        main_pathway.append(x)

    # Dropout for feature selection
    main_pathway = [Dropout(0.5)(x) for x in main_pathway]

    # Concatenate the outputs from the three groups
    main_output = tf.concat(main_pathway, axis=-1)

    # Define the branch pathway
    branch_pathway = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Add the outputs from both pathways
    added = Add()([main_output, branch_pathway])

    # Flatten the output
    flatten = tf.keras.layers.Flatten()(added)

    # Fully connected layer
    outputs = Dense(10, activation='softmax')(flatten)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()