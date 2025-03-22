import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Lambda, Conv2D, SeparableConv2D, Concatenate

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    # Max pooling layers with different scales
    x1 = MaxPooling2D(pool_size=(1, 1), strides=1)(input_layer)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=2)(input_layer)
    x3 = MaxPooling2D(pool_size=(4, 4), strides=4)(input_layer)

    # Flatten each pooling output
    x1_flat = Flatten()(x1)
    x2_flat = Flatten()(x2)
    x3_flat = Flatten()(x3)

    # Apply dropout
    x1_flat = Dropout(0.25)(x1_flat)
    x2_flat = Dropout(0.25)(x2_flat)
    x3_flat = Dropout(0.25)(x3_flat)

    # Concatenate the flattened outputs
    concat_layer = Concatenate()([x1_flat, x2_flat, x3_flat])

    # Fully connected layer and reshape to 4D tensor
    fc_layer = Dense(128, activation='relu')(concat_layer)
    reshape_layer = tf.reshape(fc_layer, (-1, 4, 4, 8))  # Assuming output of fc_layer is 128

    # Second block
    # Split the input into four groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshape_layer)

    # Process each group with separable convolutional layers
    outputs = []
    for i in range(4):
        if i == 0:
            k_size = (1, 1)
        elif i == 1:
            k_size = (3, 3)
        elif i == 2:
            k_size = (5, 5)
        else:
            k_size = (7, 7)
        
        conv_layer = SeparableConv2D(32, kernel_size=k_size, activation='relu')(split_layer[i])
        outputs.append(conv_layer)

    # Concatenate the outputs from the four groups
    concat_outputs = Concatenate(axis=-1)(outputs)

    # Flatten the output
    flatten_layer = Flatten()(concat_outputs)

    # Fully connected layer for classification
    final_layer = Dense(10, activation='softmax')(flatten_layer)

    # Define the model
    model = Model(inputs=input_layer, outputs=final_layer)

    return model

# Create and print the model summary
model = dl_model()
model.summary()