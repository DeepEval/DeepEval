import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Block 1
    # Splitting the input tensor into three groups along the channel axis
    split_tensors = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

    # Feature extraction with different kernel sizes
    conv_1x1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_tensors[0])
    conv_3x3 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_tensors[1])
    conv_5x5 = layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split_tensors[2])
    
    # Applying Dropout to reduce overfitting
    dropout = layers.Dropout(0.5)(layers.concatenate([conv_1x1, conv_3x3, conv_5x5]))

    # Block 2
    # Creating four branches
    branch_1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(dropout)
    
    branch_2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(dropout)
    branch_2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch_2)

    branch_3 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(dropout)
    branch_3 = layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(branch_3)

    branch_4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropout)
    branch_4 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch_4)

    # Concatenating outputs from all branches
    concatenated = layers.concatenate([branch_1, branch_2, branch_3, branch_4])

    # Flattening and fully connected layer
    flatten = layers.Flatten()(concatenated)
    output = layers.Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    # Constructing the model
    model = models.Model(inputs=input_tensor, outputs=output)

    return model

# Example of creating the model
model = dl_model()
model.summary()