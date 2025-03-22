# Import necessary packages from Keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Permute, Dense
from tensorflow.keras import backend as K

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model begins by obtaining the shape of the input layer and reshaping the input tensor into three groups,
    targeting a shape of (height, width, groups, channels_per_group), where groups=3 and channels_per_group=channels/groups.
    
    Next, the model swaps the third and fourth dimensions using a permutation operation to enable channel shuffling.
    
    The tensor is then reshaped back to its original input shape. Finally, the output is passed through a fully connected layer with a softmax activation for classification.
    """

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Get the input shape of the CIFAR-10 dataset
    input_shape = x_train.shape[1:]

    # Calculate the number of channels per group
    channels_per_group = input_shape[-1] // 3

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Reshape the input tensor into three groups
    reshaped_layer = Reshape((input_shape[0], input_shape[1], 3, channels_per_group))(input_layer)

    # Swap the third and fourth dimensions using a permutation operation
    permuted_layer = Permute((3, 4, 1, 2))(reshaped_layer)

    # Reshape the tensor back to its original input shape
    reshaped_back_layer = Reshape(input_shape)(permuted_layer)

    # Pass the output through a fully connected layer with a softmax activation for classification
    output_layer = Dense(10, activation='softmax')(reshaped_back_layer)

    # Construct the deep learning model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model