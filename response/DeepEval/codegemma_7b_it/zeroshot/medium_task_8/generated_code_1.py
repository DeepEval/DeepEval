from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Lambda, Input, Conv2D, MaxPooling2D, concatenate, Activation, Flatten, Dense, add
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import Sequential

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.

    The model consists of two main components: a main path and a branch path.

    The main path splits the input into three groups along the last dimension
    by encapsulating tf.split within a Lambda layer. The first group remains unchanged,
    while the second group undergoes feature extraction via a 3x3 convolutional layer. The output
    of the second group is then combined with the third group before passing through an additional 3x3 convolution.
    Finally, the outputs of all three groups are concatenated to form the output of the main path.

    The branch path employs a 1x1 convolutional layer to process the input.
    The outputs from both the main and branch paths are fused together through addition.
    The final classification result is obtained by flattening the combined output and passing it through a fully connected layer.

    Returns:
        The constructed Keras model.
    """

    # Create input layer
    input_img = Input(shape=(32, 32, 3))

    # Main path
    main_output = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_img)

    # First group in main path remains unchanged
    main_output_1 = main_output[0]

    # Second group in main path undergoes feature extraction
    main_output_2 = Conv2D(64, (3, 3), padding='same')(main_output[1])
    main_output_2 = Activation('relu')(main_output_2)
    main_output_2 = MaxPooling2D((2, 2), strides=(2, 2))(main_output_2)

    # Third group in main path undergoes feature extraction
    main_output_3 = Conv2D(64, (3, 3), padding='same')(main_output[2])
    main_output_3 = Activation('relu')(main_output_3)
    main_output_3 = MaxPooling2D((2, 2), strides=(2, 2))(main_output_3)

    # Combine outputs of second and third groups in main path
    main_output_23 = concatenate([main_output_2, main_output_3])

    # Additional 3x3 convolution in main path
    main_output_23 = Conv2D(64, (3, 3), padding='same')(main_output_23)
    main_output_23 = Activation('relu')(main_output_23)
    main_output_23 = MaxPooling2D((2, 2), strides=(2, 2))(main_output_23)

    # Concatenate outputs of all three groups in main path
    main_output = concatenate([main_output_1, main_output_23])

    # Branch path
    branch_output = Conv2D(64, (1, 1), padding='same')(input_img)
    branch_output = Activation('relu')(branch_output)

    # Fuse outputs of main and branch paths
    output = add([main_output, branch_output])
    output = Activation('relu')(output)

    # Flattening and fully connected layer for classification
    output = Flatten()(output)
    output = Dense(10, activation='softmax')(output)

    # Create model
    model = Model(inputs=input_img, outputs=output)

    return model