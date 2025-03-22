import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2)
        path1 = conv3
        path2 = conv2
        path3 = conv1

        merge_layer = Add()([path1, path2, path3])

        return merge_layer

    branch_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    block1_output = block(input_tensor=input_layer)
    block2_output = block(input_tensor=branch_path)

    merged_blocks = Concatenate()([block1_output, block2_output])
    
    flatten_layer = Flatten()(merged_blocks)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model