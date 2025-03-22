from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Reshape, Multiply, Concatenate,Flatten

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def block(input_tensor):
        
        squeeze = GlobalAveragePooling2D()(input_tensor)
        input_channels = input_tensor.shape[-1]
        excitation_1 = Dense(units=input_channels // 3, activation='relu')(squeeze)
        excitation_2 = Dense(units=input_channels, activation='sigmoid')(excitation_1)
        excitation_3 = Reshape(target_shape=(1, 1, input_channels))(excitation_2)
        output_tensor = Multiply()([input_tensor, excitation_3])

        return output_tensor

    path1 = block(input_tensor=input_layer)
    path2 = block(input_tensor=input_layer)
    concatenated = Concatenate(axis=-1)([path1, path2])
    
    flatten = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
