import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Add, Dropout, BatchNormalization, Lambda, Concatenate, SeparableConv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Main path and Branch path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        drop1 = Dropout(rate=0.5)(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(drop1)
        branch_output = conv1  # Branch path directly connects to the input

        # Add both paths
        add_layer = Add()([conv2, branch_output])
        bn = BatchNormalization()(add_layer)
        drop2 = Dropout(rate=0.5)(bn)
        flatten = Flatten()(drop2)

        return flatten

    branch_output = main_path(input_tensor=input_layer)
    
    # Block 2: Three separate paths
    def separate_paths(input_tensor):
        # Split into three groups
        inputs_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Path 1: 1x1 conv
        conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        drop1 = Dropout(rate=0.5)(conv1)

        # Path 2: 3x3 conv
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        drop2 = Dropout(rate=0.5)(conv2)

        # Path 3: 5x5 conv
        conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        drop3 = Dropout(rate=0.5)(conv3)

        # Concatenate all paths
        concat = Concatenate()(inputs=[drop1, drop2, drop3])
        flatten = Flatten()(concat)

        # Fully connected layer
        fc = Dense(units=128, activation='relu')(flatten)
        drop4 = Dropout(rate=0.5)(fc)

        # Output layer
        output = Dense(units=10, activation='softmax')(drop4)

        return output

    block2_output = separate_paths(input_tensor=branch_output)
    model = Model(inputs=input_layer, outputs=[block2_output, main_path(input_tensor=input_layer)])

    return model

model = dl_model()
model.summary()