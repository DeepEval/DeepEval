import keras
from keras import layers

def dl_model():

    input_image = keras.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    initial_conv = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(input_image)

    # Branch 1: Local features extraction
    branch_1 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(initial_conv)

    # Branch 2: Downsampling and upsampling
    branch_2 = layers.MaxPool2D()(initial_conv)
    branch_2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(branch_2)
    branch_2 = layers.UpSampling2D()(branch_2)

    # Branch 3: Downsampling and upsampling
    branch_3 = layers.MaxPool2D()(initial_conv)
    branch_3 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(branch_3)
    branch_3 = layers.UpSampling2D()(branch_3)

    # Concatenation of branches
    concat_branches = layers.concatenate([branch_1, branch_2, branch_3])

    # Final convolutional layer
    final_conv = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(concat_branches)

    # Fully connected layers
    flatten = layers.Flatten()(final_conv)
    dense_1 = layers.Dense(units=32, activation='relu')(flatten)
    dense_2 = layers.Dense(units=16, activation='relu')(dense_1)
    output = layers.Dense(units=10, activation='softmax')(dense_2)

    model = keras.Model(inputs=input_image, outputs=output)

    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])