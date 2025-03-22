import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function defines a deep learning model for image classification using the CIFAR-10 dataset.

    The model consists of two main components: a main path and a branch path. In the main path, the input is 
    split into three groups along the last dimension. The first group remains unchanged, while the second group 
    undergoes feature extraction via a 3x3 convolutional layer. The output of the second group is then combined 
    with the third group before passing through an additional 3x3 convolution. Finally, the outputs of all three 
    groups are concatenated to form the output of the main path. The branch path employs a 1x1 convolutional 
    layer to process the input. The outputs from both the main and branch paths are fused together through 
    addition. The final classification result is obtained by flattening the combined output and passing it through 
    a fully connected layer.
    """

    input_layer = keras.Input(shape=(32, 32, 3))

    # Main Path
    main_output1 = input_layer  # First group remains unchanged
    main_output2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer[:, :, :, 1:])
    main_output2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_output2)
    main_output = Concatenate()([input_layer[:, :, :, :1], main_output2, input_layer[:, :, :, 2:]])

    # Branch Path
    branch_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine Main and Branch Paths
    combined_output = Add()([main_output, branch_output])

    # Flatten the output
    flattened_output = layers.Flatten()(combined_output)

    # Fully Connected Layer
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model