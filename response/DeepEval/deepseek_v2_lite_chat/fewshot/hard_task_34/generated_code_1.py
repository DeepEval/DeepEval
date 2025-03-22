import keras
from keras.layers import Input, Conv2D, MaxPooling2D, ReLU, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    # Main Path
    def main_path_block(input_tensor):
        for _ in range(3):
            conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=ReLU)(input_tensor)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            input_tensor = pool1
        return input_tensor
    
    main_path_output = main_path_block(input_tensor=input_layer)
    
    # Branch Path
    def branch_path_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=ReLU)(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        return pool1
    
    branch_path_output = branch_path_block(input_tensor=input_layer)
    
    # Add the outputs of the main path and branch path
    add_layer = Add()([main_path_output, branch_path_output])
    
    # Flatten the output and pass it through a fully connected layer
    flatten_layer = Flatten()(add_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model