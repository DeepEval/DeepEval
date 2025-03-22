import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K

def dl_model():
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    input_layer = Input(shape=(32, 32, 3))
    
    def channel_split_branch(input_tensor):
        x = Lambda(lambda x: K.split(x, 3, axis=3))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
        dropout = Dropout(rate=0.5)(max_pooling)
        
        return dropout
    
    main_pathway_output = channel_split_branch(input_tensor)
    
    def channel_merge_branch(input_tensor):
        x = Lambda(lambda x: K.expand_dims(x, axis=3))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        
        return conv2
    
    branch_pathway_output = channel_merge_branch(input_tensor)
    
    merged_output = Add()([main_pathway_output, branch_pathway_output])
    
    flatten_layer = Flatten()(merged_output)
    dense = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model