import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Reshape, Permute, Conv2D, MaxPooling2D, Flatten, Dense


def dl_model():
    # Load and preprocess the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the input shape
    input_shape = (32, 32, 3)  # Adjusted for CIFAR-10 images
    input_layer = Input(shape=input_shape)

    # Block 1: Split and Convolution
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    x = [Conv2D(32, (1, 1), activation='relu', padding='same')(i) for i in x]  # 1/3 channels
    x = Concatenate(axis=3)(x)

    # Block 2: Channel Shuffle and Depthwise Convolution
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(x)
    x = [Reshape((-1, x.shape[2]), input_shape=[-1, input_shape[1]//3])(i) for i in x]
    x = [Permute((3, 1, 2))(i) for i in x]  # Swap channels
    x = [Conv2D(32, (3, 3), padding='same', use_depthwise=True)(i) for i in x]
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Block 3: Depthwise Separable Convolution
    x = Conv2D(64, (3, 3), padding='same')(x)

    # Branch for direct input
    branch_input = Input(shape=input_shape)
    branch_x = Conv2D(64, (1, 1), activation='relu', padding='same')(branch_input)

    # Combine outputs
    combined_features = Add()([x, branch_x])

    # Flatten and Fully Connected Layer for Classification
    x = Flatten()(combined_features)
    output_layer = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    return model

# Call the function to create and compile the model
model = dl_model()