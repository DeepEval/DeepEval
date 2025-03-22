import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the images to [0, 1] range
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)


    def main_path(input_shape):
        input_layer = Input(shape=input_shape)
        
        # Split the input channels into three groups
        x = Lambda(lambda x: tf.split(x, 3, axis=CHANNEL_AXIS))(input_layer)
        
        # Multi-scale feature extraction
        y = [Conv2D(filters=64, kernel_size=1, activation='relu')(i) for i in x]
        y.extend([Conv2D(filters=64, kernel_size=3, activation='relu')])
        y.extend([Conv2D(filters=64, kernel_size=5, activation='relu')])
        
        # Concatenate the outputs
        z = Concatenate()(y)
        
        return Model(input_layer, z, name='Main_Path')

    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)  # Assuming input images are 32x32 pixels

    # Create the main path model
    main_model = main_path(input_shape)


    def branch_path(input_shape):
        input_layer = Input(shape=input_shape)
        
        # 1x1 convolutional layer to align the number of channels
        z = Conv2D(filters=128, kernel_size=1, activation='relu')(input_layer)
        
        # Flatten and pass through two fully connected layers
        output_layer = Flatten()(z)
        output_layer = Dense(128, activation='relu')(output_layer)
        output_layer = Dense(10, activation='softmax')(output_layer)
        
        return Model(input_layer, output_layer, name='Branch_Path')

    # Define the input shape for the branch path
    branch_input_shape = (16, 16, 128)  # Assuming the branch path takes 16x16 images as input

    # Create the branch path model
    branch_model = branch_path(branch_input_shape)


    def fusion_layer(main_output, branch_output):
        return Concatenate()([main_output, branch_output])

    # Create the fusion layer model
    fusion_model = Model([main_model.output, branch_model.output], fusion_model)

    # Add the fusion layer to the main path
    model = Model(main_model.input, fusion_model)


    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

    return model