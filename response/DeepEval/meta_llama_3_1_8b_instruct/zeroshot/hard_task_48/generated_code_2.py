import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the model, assuming the input data is in (batch_size, height, width, channels)
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1: Split the input into three groups and extract features with separable convolutional layers
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    x1 = layers.SeparableConv2D(32, (1, 1), activation='relu', name='sep_conv_1x1')(x[0])
    x1 = layers.BatchNormalization()(x1)
    
    x2 = layers.SeparableConv2D(32, (3, 3), activation='relu', name='sep_conv_3x3')(x[1])
    x2 = layers.BatchNormalization()(x2)
    
    x3 = layers.SeparableConv2D(32, (5, 5), activation='relu', name='sep_conv_5x5')(x[2])
    x3 = layers.BatchNormalization()(x3)
    
    # Concatenate the outputs of the three groups
    x = layers.Concatenate()([x1, x2, x3])

    # Block 2: Four parallel branches
    path1 = layers.SeparableConv2D(32, (1, 1), activation='relu')(x)
    path1 = layers.BatchNormalization()(path1)
    
    path2 = layers.AveragePooling2D((3, 3))(x)
    path2 = layers.SeparableConv2D(32, (1, 1), activation='relu')(path2)
    path2 = layers.BatchNormalization()(path2)
    
    path3 = layers.SeparableConv2D(32, (1, 1), activation='relu')(x)
    path3 = layers.BatchNormalization()(path3)
    path3 = layers.Conv2D(32, (1, 3), padding='valid', activation='relu')(path3)
    path3 = layers.Conv2D(32, (3, 1), padding='valid', activation='relu')(path3)
    path3 = layers.BatchNormalization()(path3)
    
    path4 = layers.SeparableConv2D(32, (1, 1), activation='relu')(x)
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(path4)
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Conv2D(32, (1, 3), padding='valid', activation='relu')(path4)
    path4 = layers.Conv2D(32, (3, 1), padding='valid', activation='relu')(path4)
    path4 = layers.BatchNormalization()(path4)
    
    # Concatenate the outputs of the four paths
    x = layers.Concatenate()([path1, path2, path3, path4])

    # Flatten the output
    x = layers.Flatten()(x)

    # Output layer with softmax activation
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
print(model.summary())