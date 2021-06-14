
import tensorflow as tf
import numpy as np
import os


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)



def H(  inputs, num_filters , dropout_rate ):
    x = tf.keras.layers.BatchNormalization( epsilon=eps )( inputs )
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), use_bias=False , kernel_initializer='he_normal' )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x1 = x
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), use_bias=False , kernel_initializer='he_normal' )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.Dropout(rate=dropout_rate )(x)
    return x



def transition(inputs, num_filters , compression_factor , dropout_rate ):
    x = tf.keras.layers.BatchNormalization( epsilon=eps )(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    num_feature_maps = inputs.shape[1] # The value of 'm'

    x = tf.keras.layers.Conv2D( np.floor( compression_factor * num_feature_maps ).astype( np.int ) ,
                               kernel_size=(1, 1), use_bias=False, padding='same' , kernel_initializer='he_normal' , kernel_regularizer=tf.keras.regularizers.l2( 1e-4 ) )(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    return x



def dense_block( inputs, num_layers, num_filters, growth_rate , dropout_rate ):
    for i in range(num_layers): 
        conv_outputs = H(inputs, num_filters , dropout_rate )
        inputs = tf.keras.layers.Concatenate()([conv_outputs, inputs])
        num_filters += growth_rate 
    return inputs, num_filters



input_shape = ( 32 , 32 , 3 ) 
num_blocks = 3
num_layers_per_block = 2
growth_rate = 32
dropout_rate = 0.4
compress_factor = 0.5
eps = 1.1e-5

num_filters = 16

inputs = tf.keras.layers.Input( shape=input_shape )
x = tf.keras.layers.Conv2D( num_filters , kernel_size=( 3 , 3 ) , use_bias=False, kernel_initializer='he_normal' , kernel_regularizer=tf.keras.regularizers.l2( 1e-4 ) )( inputs )

for i in range( num_blocks ):
    x, num_filters = dense_block( x, num_layers_per_block , num_filters, growth_rate , dropout_rate )
    x = transition(x, num_filters , compress_factor , dropout_rate )

x = tf.keras.layers.GlobalAveragePooling2D()( x ) 
x = tf.keras.layers.Dense( 10 )( x ) 
outputs = tf.keras.layers.Activation( 'softmax' )( x )


model = tf.keras.models.Model( inputs , outputs )
model.compile( loss=tf.keras.losses.categorical_crossentropy ,optimizer=tf.keras.optimizers.Adam( lr=0.0001 ) ,metrics=[ 'acc' ])
model.summary()


import datetime as dt

batch_size = 64
epochs = 100

callbacks = [
  tf.keras.callbacks.TensorBoard(log_dir='./log4/{}'.format(dt.datetime.now().strftime("Res_Dense_%Y-%m-%d-%H-%M-%S")), write_images=True),
  tf.keras.callbacks.ModelCheckpoint(filepath='./Res_Dense4/cp.ckpt',
                                                    save_weights_only=True,
                                                    verbose=1)
]


model.fit( x_train , y_train , epochs=epochs , batch_size=batch_size , validation_data=( x_test , y_test ), callbacks=callbacks )


model.save('Res_Dense.h5')
results = model.evaluate(x_test, y_test, batch_size=batch_size)
print( 'Loss = {} and Accuracy = {} %'.format( results[0] , results[1] * 100 ) )




