import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Lambda, Conv2D, Concatenate

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # First block
    x1 = MaxPooling2D(pool_size=(1, 1), strides=1)(inputs)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=2)(inputs)
    x3 = MaxPooling2D(pool_size=(4, 4), strides=4)(inputs)

    # Flatten the outputs of the pooling layers
    x1_flat = Flatten()(x1)
    x2_flat = Flatten()(x2)
    x3_flat = Flatten()(x3)

    # Apply dropout
    x1_flat = Dropout(0.2)(x1_flat)
    x2_flat = Dropout(0.2)(x2_flat)
    x3_flat = Dropout(0.2)(x3_flat)

    # Concatenate the flattened outputs
    z = Concatenate()([x1_flat, x2_flat, x3_flat])

    # Transform the output into a four-dimensional tensor
    z = Reshape((1, 1, 12))(z)  # Assuming the final shape after reshape is (1, 1, 12) based on the description

    # Second block
    def split_and_process(tensor):
        # Split the tensor into four groups
        split_tensors = tf.split(tensor, num_or_size_splits=4, axis=-1)
        
        # Process each group with a separable convolutional layer
        outputs = []
        for split_tensor in split_tensors:
            output = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_tensor)
            output = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(output)
            output = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(output)
            output = Conv2D(filters=32, kernel_size=(7, 7), activation='relu')(output)
            outputs.append(output)
        
        # Concatenate the outputs
        return Concatenate()(outputs)

    # Apply the split and process function to the tensor
    y = Lambda(split_and_process)(z)

    # Flatten the output and pass it through a fully connected layer
    y = Flatten()(y)
    outputs = Dense(10, activation='softmax')(y)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()