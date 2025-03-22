import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Add
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Number of classes
    num_classes = 10

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Feature extraction branches
    branch1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(64, (1, 1))(input_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch3 = Conv2D(64, (1, 1))(input_layer)
    branch3 = Conv2D(64, (3, 3))(branch3)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)

    # Concatenate the outputs from the three branches
    concat = concatenate([branch1, branch2, branch3])

    # Final branch with a 1x1 convolution to adjust the output dimensions
    output_branch = Conv2D(num_classes, (1, 1), activation='softmax')(concat)

    # Main path (direct input path)
    main_path = input_layer

    # Fuse main path and the branches using addition
    model = Model(inputs=input_layer, outputs=Add()([main_path, output_branch]))

    # Compile the model
    model.compile(optimizer=Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

# Build and train the model
model = dl_model()
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))