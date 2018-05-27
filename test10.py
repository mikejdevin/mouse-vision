import tensorflow as tf
import numpy as np
import cv2
import time
from ctypes import windll, Structure, c_long, byref

LEARN_RATE = 0.01
BATCH_SIZE = 100
EPOCHS = 50
SAVE_NAME = 'C:\\Users\\michael\\Documents\\mproj\\test10_save.ckpt'

#global camera hooks
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
ret, test_frame = cap.read()
test_frame_shape = test_frame.shape
input_shape = [ BATCH_SIZE , test_frame_shape[0] , test_frame_shape[1] , test_frame_shape[2] ] 

#define model
def encoder(inpu):
    with tf.variable_scope('encoding'):
        # 640 x 480 x 3 -> 3 x 4 x 9 params ~3,686,400 ops
        conv1 = tf.layers.conv2d(inpu,filters=4,kernel_size=(3,3),strides=(3,3),padding='SAME',use_bias=True,activation=tf.nn.relu,name='conv1')
        
        # 214 x 160 x 4 -> 0 params
        pool1 = tf.layers.max_pooling2d(conv1,pool_size=(2,2),strides=(2,2),name='pool1')
        
        # 107 x 80 x 4 -> 4 x 6 x 9 params ~ 1,848,960 ops
        conv2 = tf.layers.conv2d(pool1,filters=6,kernel_size=(3,3),strides=(2,2),padding='SAME',use_bias=True,activation=tf.nn.relu,name='conv2')
        
        # 54 x 40 x 6 -> FC10 ~ 129,600 ops
        flat1 = tf.reshape(conv2,[-1,54*40*6])
        skip1 = tf.layers.dense(inputs=flat1, units=10, activation=tf.nn.relu,name='skip1')
        
        # 54 x 40 x 6 -> 0 params 
        pool2 = tf.layers.max_pooling2d(conv2,pool_size=(2,2),strides=(2,2),name='pool2')
        
        # 27 x 20 x 6 -> FC10 ~ 32,400 ops
        flat2 = tf.reshape(pool2,[-1,27*20*6])
        skip2 = tf.layers.dense(inputs=flat2, units=10, activation=tf.nn.relu,name='skip2')      
        
        # 27 x 20 x 6 -> 6 x 8 x 9 params ~ 483,840 ops
        conv3 = tf.layers.conv2d(pool2,filters=8,kernel_size=(3,3),strides=(2,2),padding='SAME',use_bias=True,activation=tf.nn.relu,name='conv3')
        
        # 14 x 10 x 8 -> FC10 ~ 11,200 ops
        flat3 = tf.reshape(conv3,[-1,14*10*8])
        skip3 = tf.layers.dense(inputs=flat3, units=10, activation=tf.nn.relu,name='skip3')  
        
        # 14 x 10 x 8 -> 8 x 10 x 9 params ~ 252000
        conv4 = tf.layers.conv2d(conv3,filters=10,kernel_size=(3,3),strides=(2,2),padding='SAME',use_bias=True,activation=tf.nn.relu,name='conv4')
        
        # 7 x 5 x 10 -> FC100 ~ 35,000 ops
        flat4 = tf.reshape(conv4,[-1,7*5*10])
        dens1 = tf.layers.dense(inputs=flat4, units=100, activation=tf.nn.relu,name='dens1')  
        
        # 100 -> FC 34 ~ 3,400 ops
        dens2 = tf.layers.dense(inputs=dens1, units=34, activation=tf.nn.relu,name='dens2')  
        
        # 34 + 10 + 10 + 10
        combo = tf.concat( [ dens2 , skip1 , skip2, skip3 ] , 1)       

        # 64 -> FC 36  ~ 2304 ops
        dens3 = tf.layers.dense(inputs=dens2, units=36, activation=tf.nn.relu,name='dens3')

        # 36 -> 12 FC // 2 in test
    latent = tf.layers.dense(inputs=dens3, units=2, activation=tf.nn.relu,name='latent')
    return latent

#model inputs
input_holder_left = tf.placeholder(tf.float32, input_shape , name='left_frame')
input_holder_right = tf.placeholder(tf.float32 , input_shape , name = 'right_frame')
mouse_cheat_holder = tf.placeholder(tf.int32, [ BATCH_SIZE , 2 ] , name='mouse_pos')

#model outputs
with tf.variable_scope('left'):
    lef_lat_out = encoder(input_holder_left)
with tf.variable_scope('right'):
    lat_out2 = encoder(input_holder_right)
#error
mouse_pos_err = tf.reduce_mean(tf.losses.mean_squared_error( labels=mouse_cheat_holder,predictions=lef_lat_out))


class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

    #mouse capture
def queryMousePosition():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return [ pt.x, pt.y ]

    # simult mouse and dual cam capture 
def create_batch(num = BATCH_SIZE):
    fn = 0

    
    mposes = np.zeros((num,2),dtype=np.int32)
    lfram = np.zeros((num, test_frame_shape[0] , test_frame_shape[1], 3 ),dtype=np.uint8)
    rfram = np.zeros((num, test_frame_shape[0] , test_frame_shape[1], 3 ),dtype=np.uint8)
    ticks = time.time()
    while(fn<num):
        mposes[fn,:] = queryMousePosition()
        _ , lfram[fn,:] = cap.read()
        _ , rfram[fn,:] = cap2.read()
        fn+=1
    return mposes , lfram , rfram , ( time.time() - ticks ) / num
    
def train_batch( mposes , lfram , rfram , wat_lern_rate = LEARN_RATE , ewoks =EPOCHS , resto = False ):

    #optizer
    optimiser = tf.train.AdamOptimizer(learning_rate=wat_lern_rate).minimize(mouse_pos_err)

    #setup saver
    saver = tf.train.Saver()

    #ok now go!    
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        if (resto):
            saver.restore(sess, SAVE_NAME)
        else :
            sess.run(init_op)
        ticks = time.time()
        recent_ticks = ticks
        avg_cost = 0
        for _ewok in range(ewoks):
            _, c = sess.run([optimiser, mouse_pos_err], feed_dict={ input_holder_left: lfram.astype(np.float16) , mouse_cheat_holder: mposes})
            avg_cost += c 
            old_ticks = recent_ticks
            recent_ticks = time.time()
            print("epoch:", (_ewok + 1), "cost =", c, " time: ",recent_ticks-old_ticks)
        #save training
        saver.save(sess, SAVE_NAME)
    return time.time()-ticks
    
def testit( mposes , lfram , rfram , frame_delays):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, SAVE_NAME)
  
    
    
    
# actual programage
batchsize_entered = input("choose batch size: ")
print("starting batch")
mposes , lfram , rfram , frame_delays = create_batch()
print("batch done in {}", frame_delays * BATCH_SIZE )
ewoks_entered = input("choose epochs: ")
time_tooked = train_batch( mposes , lfram , rfram )
print("trainin done in {}", time_tooked )
#_ = input("starting playback")
#testit( mposes , lfram , rfram ,frame_delays )