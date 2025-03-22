import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def main_path(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1_3)
        
        concat = Concatenate(axis=-1)([conv1_1, conv1_2, pool1])
        
        split2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(concat)
        conv2_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split2[0])
        conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2[1])
        conv2_3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split2[2])
        
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2_3)
        
        concat = Concatenate(axis=-1)([conv2_1, conv2_2, pool2])
        
        flat = Flatten()(concat)
        
        output_layer = Dense(units=10, activation='softmax')(flat)
        
        return keras.Model(inputs=input_layer, outputs=output_layer)
    
    def branch_path(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv)
        
        return conv, pool
    
    conv1, pool1 = branch_path(input_tensor=input_layer)
    conv2, pool2 = branch_path(input_tensor=conv1)
    
    fuse = Add()([pool1, pool2])
    
    flat = Flatten()(fuse)
    
    output_layer = Dense(units=10, activation='softmax')(flat)
    
    return keras.Model(inputs=input_layer, outputs=output_layer)

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])