import keras
from keras.layers import Input, Conv2D, Add, Lambda, Dropout, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Main pathway: each group undergoes a sequence of <1x1 convolution, 3x3 convolution> followed by dropout
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    drop1 = Dropout(0.2)(conv1_2)
    
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    drop2 = Dropout(0.2)(conv2_2)
    
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    drop3 = Dropout(0.2)(conv3_2)
    
    # Concatenate the outputs from the three groups to form the main pathway
    main_path = Concatenate()([drop1, drop2, drop3])
    
    # Branch pathway: process the input through a 1x1 convolution to match the output dimension of the main pathway
    conv_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    
    # Combine the outputs from both pathways using an addition operation
    adding_layer = Add()([main_path, conv_branch])
    
    # Flatten the output and pass it through a fully connected layer to complete the classification process
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model