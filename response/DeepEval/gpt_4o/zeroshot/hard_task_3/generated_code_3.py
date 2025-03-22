from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, Flatten, Add, Lambda, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels, 3 channels)
    inputs = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    # Assuming each group will have 1 channel (since CIFAR-10 has 3 channels)
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Process each group with <1x1 conv, 3x3 conv, dropout>
    processed_channels = []
    for i in range(3):
        x = Conv2D(32, (1, 1), activation='relu')(split_channels[i])
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Dropout(0.5)(x)
        processed_channels.append(x)

    # Concatenate the processed channels to form the main pathway
    main_pathway = Concatenate(axis=-1)(processed_channels)

    # Branch pathway: 1x1 convolution to match output dimension of the main pathway
    branch_pathway = Conv2D(192, (1, 1), activation='relu')(inputs)  # 64*3 = 192

    # Combine both pathways using an addition operation
    combined = Add()([main_pathway, branch_pathway])

    # Flatten and fully connected layer for classification
    x = Flatten()(combined)
    outputs = Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model