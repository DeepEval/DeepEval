import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, MaxPooling2D, Dense, BatchNormalization, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def block1(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensor[0])
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv1_2)

        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensor[1])
        conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2_1)
        conv2_3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv2_2)

        conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensor[2])
        conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv3_1)
        conv3_3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv3_2)
        
        output_tensor = Concatenate()([conv1_3, conv2_3, conv3_3])

        return output_tensor

    block1_output = block1(input_layer)
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(block1_output)

    def block2(input_tensor):
        global_pool = MaxPooling2D(pool_size=(8, 8))(input_tensor)
        
        fc1 = Dense(units=64)(global_pool)
        fc2 = Dense(units=64, activation='relu')(fc1)
        weights = Dense(units=64, activation='relu')(fc2)
        weights = tf.reshape(weights, (-1, 1, 1, 64))
        output_tensor = input_tensor * weights

        return output_tensor

    main_path_output = block2(transition_conv)

    branch_output = input_layer
    
    combined_output = main_path_output + branch_output

    output_layer = Dense(units=10, activation='softmax')(combined_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model