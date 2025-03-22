import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Lambda, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Splitting input into three groups
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main pathway processing each group
    def main_pathway(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        dropout = Dropout(rate=0.5)(conv2)
        return dropout
    
    main_outputs = [main_pathway(group) for group in inputs_groups]
    main_path = Concatenate()(main_outputs)

    # Branch pathway processing input
    branch_path = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combining main and branch pathways
    combined = Add()([main_path, branch_path])

    # Fully connected layer for classification
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model