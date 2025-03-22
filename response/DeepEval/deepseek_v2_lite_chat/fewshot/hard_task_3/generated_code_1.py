import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dropout, Add, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Main pathway
    def main_path(input_tensor):
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor[0])
        conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_tensor[1])
        conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input_tensor[2])
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv3_1)
        drop1 = Dropout(rate=0.2)(pool1)
        return drop1
    
    # Branch pathway
    def branch_path(input_tensor):
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        return conv1_1
    
    # Combine outputs from both pathways
    main_output = main_path(input_tensor=split_layer)
    branch_output = branch_path(input_tensor=split_layer[2])
    add_layer = Add()([main_output, branch_output])
    
    # Fully connected layer
    flatten_layer = Flatten()(add_layer)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    drop2 = Dropout(rate=0.5)(dense1)
    output_layer = Dense(units=10, activation='softmax')(drop2)
    
    # Return constructed model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model