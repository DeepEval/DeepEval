import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def dl_model():
    # Define input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Define first block
    def block1(input_tensor):
        conv = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        drop = layers.Dropout(0.2)(conv)
        conv_restored = layers.Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop)
        main_path = conv_restored
        
        branch_path = input_tensor
        output_block = layers.Add()([main_path, branch_path])
        return output_block

    block1_output = block1(input_layer)

    # Define second block
    def block2(input_tensor):
        x = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        group1 = layers.SeparableConv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(x[0])
        group2 = layers.SeparableConv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x[1])
        group3 = layers.SeparableConv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(x[2])
        group1_drop = layers.Dropout(0.2)(group1)
        group2_drop = layers.Dropout(0.2)(group2)
        group3_drop = layers.Dropout(0.2)(group3)
        concat_output = layers.Concatenate()([group1_drop, group2_drop, group3_drop])
        return concat_output

    block2_output = block2(block1_output)

    # Define flatten layer and fully connected layer
    flatten_layer = layers.Flatten()(block2_output)
    dense_layer = layers.Dense(10, activation='softmax')(flatten_layer)

    # Construct model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    return model

model = dl_model()
model.summary()