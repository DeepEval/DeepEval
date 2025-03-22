import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three channels
    split_tensor = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Process each channel group
    group1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_tensor[0])
    group1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group1)
    group1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(group1)
    
    group2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_tensor[1])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group2)
    group2 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(group2)
    
    group3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_tensor[2])
    group3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group3)
    group3 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(group3)

    # Combine outputs from groups
    main_path = Add()([group1, group2, group3])
    
    # Fuse with original input
    output = Add()([main_path, input_layer])

    # Flatten and classify
    flatten_layer = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model