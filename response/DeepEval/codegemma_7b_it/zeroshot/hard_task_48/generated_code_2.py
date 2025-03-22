import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1
    split_input = layers.Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(inputs)
    conv_outputs = []
    for filters, kernel_size in [(256, 1), (256, 3), (256, 5)]:
        conv = layers.Conv2D(filters, kernel_size, padding='same')(split_input)
        bn = layers.BatchNormalization()(conv)
        conv_outputs.append(bn)

    concat_block1 = layers.concatenate(conv_outputs)

    # Block 2
    path1 = layers.Conv2D(256, 1, padding='same')(concat_block1)
    path2 = layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(concat_block1)
    path2 = layers.Conv2D(256, 1, padding='same')(path2)
    path3_a = layers.Conv2D(256, 1, padding='same')(concat_block1)
    path3_b = layers.Conv2D(256, (1, 3), padding='same')(path3_a)
    path3_c = layers.Conv2D(256, (3, 1), padding='same')(path3_a)
    concat_path3 = layers.concatenate([path3_b, path3_c])
    path4_a = layers.Conv2D(256, 1, padding='same')(concat_block1)
    path4_b = layers.Conv2D(256, 3, padding='same')(path4_a)
    concat_path4 = layers.concatenate([path4_a, path4_b])

    # Concatenate all paths
    concat_block2 = layers.concatenate([path1, path2, concat_path3, concat_path4])

    # Final layers
    flatten = layers.Flatten()(concat_block2)
    outputs = layers.Dense(10, activation='softmax')(flatten)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])