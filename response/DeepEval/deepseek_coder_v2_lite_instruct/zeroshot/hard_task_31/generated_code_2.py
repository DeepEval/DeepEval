import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block
    # Main path
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    
    # Branch path
    branch = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    
    # Add outputs from both paths
    x = Add()([x, branch])

    # Second block
    # Split the input into three groups
    split_points = [1, 2]  # This splits the last dimension into three groups
    split_tensors = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=split_points, axis=-1))(x)
    
    # Apply separable convolutions to each group with different kernel sizes
    conv1x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_tensors[0])
    conv3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_tensors[1])
    conv5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_tensors[2])
    
    # Apply dropout to each convolution
    conv1x1 = Dropout(0.2)(conv1x1)
    conv3x3 = Dropout(0.2)(conv3x3)
    conv5x5 = Dropout(0.2)(conv5x5)
    
    # Concatenate the outputs from the three groups
    x = Add()([conv1x1, conv3x3, conv5x5])

    # Flatten the output and add a fully connected layer for classification
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()