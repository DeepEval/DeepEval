import keras
from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense, Concatenate
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator


def dl_model():
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # One-hot encode the labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Input shape
    input_shape = (32, 32, 3)

    # Create the base model from VGG16, excluding the last three layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Prevent the base model from being trainable
    for layer in base_model.layers:
        layer.trainable = False

    # # Feature extraction layer
    # f = base_model.output

    # # Add a fully-connected layer
    # f = Flatten()(f)

    # # Add a fully-connected layer
    # f = Dense(1024, activation='relu')(f)

    # # Add a dropout layer
    # f = Dropout(0.5)(f)

    # # Output layer
    # predictions = Dense(10, activation='softmax')(f)

    # # Define the Keras model
    # model = Model(inputs=base_model.input, outputs=predictions)

    # # Freeze the base model
    # for layer in model.layers[:-4]:
    #     layer.trainable = False

    # Define the paths for the main and branch paths
    main_path_input = Input(shape=input_shape)
    branch_path_input = Input(shape=(1024,))

    # Split the input into three groups along the channel
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(main_path_input)

    # Extract features using depthwise separable convolutional layers
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x[0])
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x[1])
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x[2])

    # Concatenate the outputs from the three groups
    main_path_output = Concatenate()([x[0], x[1], x[2]])

    # Align the number of output channels in the branch path with the main path
    branch_path_output = Conv2D(64, (1, 1), activation='relu', padding='same')(branch_path_input)

    # Add the outputs from the main and branch paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten and apply two fully connected layers
    combined_output = Flatten()(combined_output)
    combined_output = Dense(512, activation='relu')(combined_output)
    predictions = Dense(10, activation='softmax')(combined_output)

    # Define the complete Keras model
    model = Model(inputs=[main_path_input, branch_path_input], outputs=predictions)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Instantiate and return the model
model = dl_model()
model.summary()