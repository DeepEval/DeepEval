import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Lambda, Flatten, Dense, Input

def dl_model():
    model = Sequential()

    # Input layer
    input_tensor = Input(shape=(32, 32, 3))

    # Split the channels
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_tensor)

    # Apply different kernel sizes
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensor[0])
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split_tensor[1])
    conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(split_tensor[2])

    # Concatenate the outputs
    merged_features = tf.concat([conv1x1, conv3x3, conv5x5], axis=3)

    # Flatten and dense layers
    x = Flatten()(merged_features)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    
    return model