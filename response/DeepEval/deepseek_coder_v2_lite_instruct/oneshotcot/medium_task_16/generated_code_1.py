import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Lambda, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group
    conv_groups = [Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split) for split in split_layer]

    # Downsample each group via average pooling
    pooled_groups = [AveragePooling2D(pool_size=(8, 8))(conv) for conv in conv_groups]

    # Concatenate the three groups along the channel dimension
    concatenated_features = Concatenate(axis=-1)(pooled_groups)

    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened_features = Flatten()(concatenated_features)

    # Pass through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model (optional, depending on the use case)
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])