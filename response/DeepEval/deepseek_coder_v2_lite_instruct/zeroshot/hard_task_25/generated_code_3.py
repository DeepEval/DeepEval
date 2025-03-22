import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main Path
    x = Conv2D(64, (1, 1), activation='relu')(inputs)  # 1x1 convolutional layer

    # First Branch (local features)
    branch1 = Conv2D(32, (3, 3), activation='relu')(x)  # 3x3 convolutional layer

    # Second Branch (downsample and process)
    branch2 = AveragePooling2D((2, 2))(x)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)

    # Third Branch (downsample and process)
    branch3 = AveragePooling2D((2, 2))(x)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)

    # Concatenate outputs of all branches
    combined = Concatenate()([branch1, branch2, branch3])
    main_path_output = Conv2D(64, (1, 1), activation='relu')(combined)  # 1x1 convolutional layer

    # Branch Path
    branch_path_output = Conv2D(64, (1, 1), activation='relu')(inputs)  # 1x1 convolutional layer

    # Fuse Main Path and Branch Path
    fused_output = tf.add(main_path_output, branch_path_output)

    # Flatten and apply fully connected layer for classification
    flattened = Flatten()(fused_output)
    outputs = Dense(10, activation='softmax')(flattened)  # 10-class classification

    # Construct and return the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage:
# model = dl_model()
# model.summary()