from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

def dl_model():
    # Input layer
    img_input = layers.Input(shape=(32, 32, 3))

    # Split the image into three groups along the channel dimension
    img_split = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(img_input)

    # Apply different convolutional kernels to each group
    conv_1x1 = layers.Lambda(lambda x: layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x))(img_split[0])
    conv_3x3 = layers.Lambda(lambda x: layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x))(img_split[1])
    conv_5x5 = layers.Lambda(lambda x: layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x))(img_split[2])

    # Concatenate the outputs from the three groups
    concat = layers.concatenate([conv_1x1, conv_3x3, conv_5x5])

    # Flatten the concatenated features
    flat = layers.Flatten()(concat)

    # Fully connected layers
    dense = layers.Dense(128, activation='relu')(flat)
    output = layers.Dense(10, activation='softmax')(dense)

    # Create the model
    model = models.Model(img_input, output)

    return model

# Compile the model
model = dl_model()
model.compile(optimizer=optimizers.Adam(), loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])