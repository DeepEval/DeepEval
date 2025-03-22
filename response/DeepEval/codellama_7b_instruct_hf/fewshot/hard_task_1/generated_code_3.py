import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Activation, Add, Lambda, Reshape, Concatenate, DepthwiseConv2D
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    block1_input = Input(shape=input_shape)
    block1_conv = Conv2D(32, (3, 3), activation='relu')(block1_input)
    block1_gap = GlobalAveragePooling2D()(block1_conv)
    block1_gmp = GlobalMaxPooling2D()(block1_conv)
    block1_fc1 = Dense(64, activation='relu')(block1_gap)
    block1_fc2 = Dense(64, activation='relu')(block1_gmp)
    block1_output = Add()([block1_fc1, block1_fc2])

    # Define the second block
    block2_input = Input(shape=input_shape)
    block2_conv = Conv2D(32, (3, 3), activation='relu')(block2_input)
    block2_gap = GlobalAveragePooling2D()(block2_conv)
    block2_gmp = GlobalMaxPooling2D()(block2_conv)
    block2_fc1 = Dense(64, activation='relu')(block2_gap)
    block2_fc2 = Dense(64, activation='relu')(block2_gmp)
    block2_output = Add()([block2_fc1, block2_fc2])

    # Define the third block
    block3_input = Input(shape=input_shape)
    block3_conv = Conv2D(32, (3, 3), activation='relu')(block3_input)
    block3_gap = GlobalAveragePooling2D()(block3_conv)
    block3_gmp = GlobalMaxPooling2D()(block3_conv)
    block3_fc1 = Dense(64, activation='relu')(block3_gap)
    block3_fc2 = Dense(64, activation='relu')(block3_gmp)
    block3_output = Add()([block3_fc1, block3_fc2])

    # Define the fourth block
    block4_input = Input(shape=input_shape)
    block4_conv = Conv2D(32, (3, 3), activation='relu')(block4_input)
    block4_gap = GlobalAveragePooling2D()(block4_conv)
    block4_gmp = GlobalMaxPooling2D()(block4_conv)
    block4_fc1 = Dense(64, activation='relu')(block4_gap)
    block4_fc2 = Dense(64, activation='relu')(block4_gmp)
    block4_output = Add()([block4_fc1, block4_fc2])

    # Define the final classification layer
    final_input = Input(shape=input_shape)
    final_conv = Conv2D(32, (3, 3), activation='relu')(final_input)
    final_gap = GlobalAveragePooling2D()(final_conv)
    final_fc = Dense(10, activation='softmax')(final_gap)
    final_output = final_fc

    # Define the model
    model = Model(inputs=final_input, outputs=final_output)

    return model