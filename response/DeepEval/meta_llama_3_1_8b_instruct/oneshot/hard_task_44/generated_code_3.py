import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_layer = keras.Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(64, (5, 5), padding='same', activation='relu')(input_layer)
    split_input = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(conv1)
    output1 = layers.Concatenate()([conv1, conv2, conv3])
    output1 = layers.Dropout(0.2)(output1)
    
    def block1(input_tensor):
        split_input = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(128, (1, 1), padding='same', activation='relu')(split_input[0])
        conv2 = Conv2D(128, (3, 3), padding='same', activation='relu')(split_input[1])
        conv3 = Conv2D(128, (5, 5), padding='same', activation='relu')(split_input[2])
        maxpool = Conv2D(128, (1, 1), padding='same', activation='relu')(layers.MaxPooling2D(pool_size=(1, 1))(input_tensor))
        output_tensor = layers.Concatenate()([conv1, conv2, conv3, maxpool])
        return output_tensor

    block_output = block1(output1)
    
    # Block 2
    conv1_1 = Conv2D(256, (1, 1), padding='same', activation='relu')(block_output)
    conv1_2 = Conv2D(256, (1, 1), padding='same', activation='relu')(block_output)
    conv2_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(block_output)
    conv2_2 = Conv2D(256, (5, 5), padding='same', activation='relu')(block_output)
    maxpool1 = Conv2D(256, (3, 3), padding='same', activation='relu')(layers.MaxPooling2D(pool_size=(3, 3))(block_output))
    maxpool1 = Conv2D(256, (1, 1), padding='same', activation='relu')(maxpool1)
    
    def block2(input_tensor):
        conv1 = Conv2D(512, (1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(512, (1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(512, (3, 3), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(512, (5, 5), padding='same', activation='relu')(input_tensor)
        maxpool = Conv2D(512, (1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = layers.Concatenate()([conv1, conv2, conv3, conv4, maxpool])
        return output_tensor

    block_output2 = block2(layers.Concatenate()([conv1_1, conv2_1, maxpool1]))
    
    # Flatten and Dense
    bath_norm = layers.BatchNormalization()(block_output2)
    flatten_layer = layers.Flatten()(bath_norm)
    dense1 = layers.Dense(1024, activation='relu')(flatten_layer)
    output_layer = layers.Dense(10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()