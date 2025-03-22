import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D, Lambda
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    dropout1 = keras.layers.Dropout(rate=0.5)(max_pool1)
    
    # Branch Path
    branch_conv = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Add Block
    def add_block(input_tensor):
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(path2)
        add_tensor = Concatenate()([path2, input_tensor])
        return add_tensor
    
    add_output = add_block(dropout1)
    batch_norm_add = BatchNormalization()(add_output)
    
    # Block 2: Three Groups
    def split_groups(input_tensor):
        split = Lambda(lambda x: tf.split(x, [16, 16, 16], axis=-1))(input_tensor)
        conv_groups = [
            SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(group) for group in split
        ]
        conv_groups += [
            keras.layers.Dropout(rate=0.5)(group) for group in conv_groups
        ]
        return Concatenate()(conv_groups)
    
    group1 = split_groups(batch_norm_add)
    group2 = split_groups(branch_conv)
    group3 = split_groups(dropout1)
    
    # Final Output
    flatten_layer = Flatten()(group3)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])