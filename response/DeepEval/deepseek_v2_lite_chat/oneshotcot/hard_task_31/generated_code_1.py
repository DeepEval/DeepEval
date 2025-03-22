import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, multiply
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv_dropout_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    dropout_layer = Dense(units=4*32, activation='sigmoid')(conv_dropout_layer)
    conv_dropout_layer_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='linear', padding='same')(dropout_layer)
    concat_layer = Concatenate()([conv_dropout_layer_2, input_layer])
    
    # Branch path
    branch_input = input_layer
    conv_layer_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(branch_input)
    conv_layer_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch_input)
    conv_layer_3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(branch_input)
    dropout_layer_1 = Dense(units=4*64, activation='sigmoid')(conv_layer_3)
    conv_layer_4 = Conv2DTranspose(filters=64, kernel_size=(1, 1), activation='linear', padding='same')(dropout_layer_1)
    concat_branch_path = Concatenate()([conv_layer_4, conv_layer_3, conv_layer_2, conv_layer_1])
    
    # Add branch path outputs to main path outputs
    combined_outputs = Concatenate()([concat_layer, concat_branch_path])
    
    # First block
    block1_output = combined_outputs
    block1_output = BatchNormalization()(block1_output)
    block1_output = MaxPooling2D(pool_size=(2, 2))(block1_output)
    
    # Second block
    split_lambda = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))
    split_output = split_lambda(block1_output)
    group_1_output = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same')(split_output[0])
    group_2_output = Conv2D(filters=128, kernel_size=(3x3, 1), activation='relu', padding='valid')(split_output[1])
    group_3_output = Conv2D(filters=128, kernel_size=(1, 3), activation='relu', padding='valid')(split_output[2])
    concat_group_outputs = Concatenate()([group_1_output, group_2_output, group_3_output])
    block2_output = BatchNormalization()(concat_group_outputs)
    block2_output = Flatten()(block2_output)
    dense_layer_1 = Dense(units=256, activation='relu')(block2_output)
    dense_layer_2 = Dense(units=128, activation='relu')(dense_layer_1)
    output_layer = Dense(units=10, activation='softmax')(dense_layer_2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model