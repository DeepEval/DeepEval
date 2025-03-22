# Import necessary libraries
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path and a branch path. The main path processes the input features 
    by a 1x1 initial convolutional layer, which then branches into three distinct paths. 
    Branch 1 applies a 3x3 convolutional layer for feature extraction. Meanwhile, Branches 2 and 3 
    downsample the feature maps using max pooling layers, apply 3x3 convolutional layers, and 
    subsequently upsample the feature maps back to their original size. The outputs from all 
    branches are concatenated and passed through another 1x1 convolutional layer to produce the 
    final output of the main path. The branch path also begins with a 1x1 convolutional layer, 
    ensuring that the number of channels matches that of the main path. The outputs from both paths 
    are then added and processed through two fully connected layers to perform classification 
    across 10 classes.
    """

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # Branch 1
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Branch 2
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x2 = MaxPooling2D((2, 2), strides=2)(x2)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x2 = UpSampling2D((2, 2))(x2)

    # Branch 3
    x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x3 = MaxPooling2D((2, 2), strides=2)(x3)
    x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
    x3 = UpSampling2D((2, 2))(x3)

    # Concatenate the outputs from all branches
    x = concatenate([x1, x2, x3])
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)

    # Define the branch path
    branch_inputs = Input(shape=input_shape)
    branch_x = Conv2D(32, (1, 1), activation='relu', padding='same')(branch_inputs)

    # Concatenate the branch path with the main path
    x = concatenate([x, branch_x])

    # Define the output layers
    x = Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[inputs, branch_inputs], outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model