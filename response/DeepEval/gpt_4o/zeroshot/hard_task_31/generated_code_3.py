import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image dimensions
    num_classes = 10  # Number of classes in CIFAR-10

    # Input layer
    inputs = Input(shape=input_shape)

    # First Block
    # Main Path
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)  # Increase width
    x = Dropout(0.2)(x)  # Dropout
    x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)  # Restore channel count

    # Branch Path
    branch = inputs  # Direct connection

    # Add outputs from both paths
    block1_output = Add()([x, branch])

    # Second Block
    # Split input into 3 parts along the last dimension
    def split_tensor(tensor):
        return tf.split(tensor, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_tensor)(block1_output)
    
    # 1x1 Separable Convolution
    conv1 = SeparableConv2D(32, (1, 1), activation='relu', padding='same')(split_layer[0])
    conv1 = Dropout(0.2)(conv1)

    # 3x3 Separable Convolution
    conv2 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(split_layer[1])
    conv2 = Dropout(0.2)(conv2)

    # 5x5 Separable Convolution
    conv3 = SeparableConv2D(32, (5, 5), activation='relu', padding='same')(split_layer[2])
    conv3 = Dropout(0.2)(conv3)

    # Concatenate outputs
    block2_output = Concatenate()([conv1, conv2, conv3])

    # Output Layer
    x = Flatten()(block2_output)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Construct the Model
    model = Model(inputs, outputs)

    return model

# Create the model
model = dl_model()