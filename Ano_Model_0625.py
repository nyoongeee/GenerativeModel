from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import csv
from skimage.measure import compare_ssim as ssim
#import tf.image_ssim_multiscale as ssim_multi

from ops import *
from utils import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=False,
        batch_size=64, sample_num = 64, output_height=64, output_width=64,
        z_dim=100, gf_dim=64, df_dim=64,
        gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
        input_fname_pattern='*.jpg', test_batch_size = 1, checkpoint_dir=None, 
        sample_dir=None, test_dir = None, saved_file_name = None, train_dir= None):
        """
        Args:
        sess: TensorFlow session
        batch_size: The size of batch. Should be specified before training.
        z_dim: (optional) Dimension of dim for Z. [100]
        gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
        df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
        gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
        dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
    
        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.test_dir = os.path.join('./',test_dir)
        self.saved_file_name = saved_file_name

        self.data = glob(os.path.join(train_dir, '*.npy'))
        self.c_dim = 3

        self.build_model()

    def build_model(self):

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        #placeholders
        self.y = None

        if self.crop: #for training
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else: #for test
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        #Construct Generator and Discriminators
        self.G                  = self.generator(self.z, self.y)
        self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)

        self.sampler            = self._sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
        #summary op.
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        #Create Loss Functions
        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        #summary op.
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    
        #total loss
        self.d_loss = self.d_loss_real + self.d_loss_fake

        #summary op.
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=None)
        
    #def _gram_matrix(self, x, area, depth):
                
    #    x = tf.reshape(x, (area, depth))
    #    gram = tf.matmul(tf.transpose(x), x)
    #    return gram

    def _gram_matrix(self, tensor, shape=None):

        if shape is not None:
            B = shape[0]  # batch size
            HW = shape[1] # height x width
            C = shape[2]  # channels
            CHW = C*HW
        else:
            B, H, W, C = map(lambda i: i.value, tensor.get_shape())
            HW = H*W
            CHW = W*H*C

        # reshape the tensor so it is a (B, 2-dim) matrix
        # so that 'B'th gram matrix can be computed
        feats = tf.reshape(tensor, (B, HW, C))

        # leave dimension of batch as it is
        feats_T = tf.transpose(feats, perm=[0, 2, 1])

        # paper suggests to normalize gram matrix by its number of elements
        gram = tf.matmul(feats_T, feats) / CHW
        #print(gram.shape)
        return gram
    
    def gram_matrix_loss(self, x, y):
        _, h, w, d = x.shape
        area = int(h*w)
        depth = int(d)
        gram_generate =  self._gram_matrix(x)
        gram_test = self._gram_matrix(y)
        #gram_loss = (1 / (4 * depth**2 * area**2)) * tf.reduce_sum(tf.pow(gram_generate - gram_test, 2), axis = [1,2])
        gram_loss = tf.reduce_sum(tf.pow(gram_generate - gram_test, 2), axis = [1,2])
        #print(gram_loss.shape)
        return gram_loss
    
    def ssim_loss(self, x ,y):
        batch_size = x.shape[0]
        ssim_list = []
        for _idx in range(0, batch_size):
            img_x = np.squeeze(x[_idx,:,:,:])
            img_y = np.squeeze(y[_idx,:,:,:])
            input_min = np.fmin(img_x.min(), img_y.min())
            input_max = np.fmax(img_x.max(), img_y.max())
            #ssim_score = ssim(img_x, img_y, multichannel=True, data_range = input_max - input_min)
            ssim_score = ssim(img_x, img_y, multichannel=True, data_range = input_max - input_min, gaussian_weights = True, 
                              sigma = 1.5, use_sample_covariance = False, K1= 0.001 * (2^7) , K2 = 0.001  * (2^7))
            ssim_list.append(ssim_score)
        #print(ssim_list)
        return ssim_list
    
    
    def ssim_evaluate_in_batch(self, x ,y):
        batch_size = x.shape[0]
        ssim_list = []
        for _idx in range(0, batch_size):
            img_x = np.squeeze(x[_idx,:,:,:])
            img_y = np.squeeze(y[_idx,:,:,:])
            input_min = np.fmin(img_x.min(), img_y.min())
            input_max = np.fmax(img_x.max(), img_y.max())
            #ssim_score = ssim(img_x, img_y, multichannel=True, data_range = input_max - input_min)
            ssim_score = ssim(img_x, img_y, multichannel=True, data_range = input_max - input_min, gaussian_weights = True, 
                              sigma = 1.5, use_sample_covariance = False, K1= 0.001 * (2^7) , K2 = 0.001  * (2^7))
            ssim_list.append(ssim_score)
        #rint(ssim_list)
        return ssim_list
    
    def train(self, config):
        #d_optim = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        d_optim = tf.train.GradientDescentOptimizer(config.d_learning_rate).minimize(self.d_loss, var_list=self.d_vars)        
        g_optim = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        #summary_op: merge summary
        self.g_sum = merge_summary([self.z_sum, self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        #Create Tensorboard
        #self.writer = SummaryWriter("./logs", self.sess.graph)
        os.makedirs('./logs/{}'.format(self.dataset_name), exist_ok=True)
        self.writer = SummaryWriter('./logs/{}'.format(self.dataset_name), self.sess.graph)

        #Create Sample Benchmarks for monitoring of train results: use same random noises and real-images
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
        sample_files = self.data[0:self.sample_num] #name_list
        sample = [ np.load(sample_file) for sample_file in sample_files]

        sample_inputs = np.array(sample).astype(np.float32)
  
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size
        sample_feed_dict = {self.z: sample_z, self.inputs: sample_inputs}
        
        print('*' *10 + 'Training Start' + '*'* 10)
        for epoch in xrange(config.epoch):
            for idx in xrange(0, batch_idxs):
                #Prepare batch data for learning
                batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [ np.load(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                #Prepare batch random noises for learning
                #batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
                
                # Sample from gaussian distribution
                batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                #Make feed dictionary
                d_feed_dict = {self.inputs: batch_images, self.z: batch_z}
                d_fake_feed_dict  = {self.z: batch_z}
                d_real_feed_dict  = {self.inputs: batch_images}
                g_feed_dict = {self.z:batch_z}

                #Run Optimization and Summary Operation of Discriminator
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict = d_feed_dict)
                self.writer.add_summary(summary_str, counter)

                #Run Optimization and Summary Operation of Generator
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict = g_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict = g_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Calculate Loss Values of Discriminator and Generator

                errD_fake = self.d_loss_fake.eval(feed_dict = d_fake_feed_dict)
                errD_real = self.d_loss_real.eval(feed_dict = d_real_feed_dict)
                errG = self.g_loss.eval(feed_dict = g_feed_dict)

                counter += 1

                if np.mod(counter, 100) == 1:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, d_fake: %.8f, d_real: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG, errD_fake, errD_real))

                if np.mod(counter, 1000) == 1:
                    samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss], feed_dict = sample_feed_dict)
                    ## 3 lines below were makred # before jang touched ##
#                     save_images(samples, image_manifold_size(samples.shape[0]), './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    os.makedirs('./{}/{}'.format(config.sample_dir,self.dataset_name), exist_ok=True) 
                    np.save('./{}/{}/train_{:02d}_{:04d}.npy'.format(config.sample_dir,self.dataset_name, epoch, idx), samples)
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 

                if np.mod(counter, 1000) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False, batch_size = None):
        if batch_size == None: batch_size = self.batch_size
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = leak_relu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = leak_relu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = leak_relu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = leak_relu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            #h4 = linear(tf.contrib.layers.flatten(h3),1,'d_h4_lin')
            h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def feature_match_layer(self, image, y=None, reuse=False, batch_size = None, match_layer='h3'):
        if batch_size == None: batch_size = self.batch_size
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = leak_relu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = leak_relu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = leak_relu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = leak_relu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            
            if match_layer == 'h1':
                return h1
            elif match_layer == 'h2':
                return h2
            else:
                return h3

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = leak_relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = leak_relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = leak_relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = leak_relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def _sampler(self, z, y=None, batch_size = None):
        if batch_size == None:
            batch_size = self.batch_size

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'), [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = leak_relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = leak_relu(self.g_bn1(h1, train=False))                

            h2 = deconv2d(h1, [batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = leak_relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = leak_relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [batch_size, s_h, s_w, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)

    def get_test_data(self):
        self.test_data_names = sorted(glob(self.test_dir+'/*.npy'))

    def anomaly_detector(self, ano_para=0.5, dis_loss='feature', matching_layer_name='h3'):
        self.get_test_data()

        #with variable_scope("anomaly_detector"):
        self.y = None

        if self.crop: 
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else: #for test
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.test_inputs = tf.placeholder(tf.float32, [self.test_batch_size] + image_dims, name='test_images')
        test_inputs = self.test_inputs

        # Anomaly initialization from uniform distribution
        #self.ano_z = tf.get_variable('ano_z', shape = [1, self.z_dim], dtype = tf.float32, 
        #    initializer = tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32))

        # Anomaly initialization from normal distribution
        self.ano_z = tf.get_variable('ano_z', shape = [self.test_batch_size, self.z_dim], dtype = tf.float32, 
            initializer = tf.random_normal_initializer(mean=0, stddev=1, dtype=tf.float32))
        
        self.ano_y = None

        self.ano_G = self._sampler(self.ano_z, self.ano_y, batch_size=self.test_batch_size)

        #Original residual loss
        #self.res_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(test_inputs, self.ano_G))))
        # Mean squared loss
        #self.res_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(test_inputs, self.ano_G),2)))
        #self.res_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(test_inputs, self.ano_G),2)))
        #self.res_loss = tf.reduce_mean(tf.abs(tf.subtract(test_inputs, self.ano_G)), axis = [1,2,3])
        self.res_loss = tf.abs(tf.reduce_mean(tf.subtract(test_inputs, self.ano_G), axis = [1,2,3]))
        #print(self.res_loss)
        
        #Residual loss with SSIM
        #self.ssim_score = tf.image.ssim_multiscale(test_inputs, self.ano_G, max_val=2.0)
        #self.res_loss = tf.reduce_mean(tf.squeeze(tf.image.ssim_multiscale(test_inputs, self.ano_G, max_val=2.0)))

        #Create Anomaly Score 
        if dis_loss == 'feature': # if discrimination loss with feature matching (same with paper)
            self.dis_f_z = self.feature_match_layer(self.ano_G, self.ano_y, reuse=True, 
                                                    batch_size=self.test_batch_size, match_layer= matching_layer_name)
            self.dis_f_input = self.feature_match_layer(test_inputs, self.ano_y, reuse=True,
                                                        batch_size=self.test_batch_size, match_layer= matching_layer_name)
            self.dis_loss = tf.reduce_mean(tf.abs(tf.subtract(self.dis_f_z, self.dis_f_input)), axis = [1,2,3])
            
            # Gram matrix loss (aka style loss)
            #self.dis_loss = self.gram_matrix_loss(self.dis_f_z, self.dis_f_input)
        else: # if dis_loss with original generator's loss in  DCGAN
            test_D, test_D_logits_ = self.discriminator(self.ano_G, self.ano_y, reuse=True, batch_size=self.test_batch_size)
            self.dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = test_D_logits_, labels = tf.ones_like(test_D)))
            
        # Gram matrix loss (aka style loss)
        #self.anomaly_score = self.res_loss + self.dis_loss
        self.anomaly_score = (1. - ano_para)* self.res_loss + ano_para* self.dis_loss
        #print(self.anomaly_score)
        
        self.gram_score_h2 = self.gram_matrix_loss(self.feature_match_layer(self.ano_G, self.ano_y, reuse=True, 
                                                    batch_size=self.test_batch_size, match_layer= 'h2'),
                              self.feature_match_layer(test_inputs, self.ano_y, reuse=True,
                                                        batch_size=self.test_batch_size, match_layer= 'h2'))
        self.gram_score = self.gram_matrix_loss(self.dis_f_z, self.dis_f_input)

        t_vars = tf.trainable_variables()
        self.z_vars = [var for var in t_vars if 'ano_z' in var.name]
        #print(test_inputs, self.ano_G, dis_f_z, dis_f_input)

    def train_anomaly_detector(self, config, test_data_list, subset_idx = None):
        #print("Filename: ", test_data_name, "Anomaly is detecting")
        #print(np.shape(test_data))
        #self.sess.run(self.ano_z.initializer)
        z_optim = tf.train.AdamOptimizer(config.test_learning_rate, beta1=config.beta1).minimize(self.anomaly_score, var_list = self.z_vars)
        initialize_uninitialized(self.sess)
       
        batch_idxs = len(test_data_list) // config.test_batch_size
        print('*' *10 + 'Training Start' + '*'* 10)
        for idx in xrange(0, batch_idxs):
            #start_t = time.time()
            #Prepare batch data for learning
            batch_files = test_data_list[idx*config.test_batch_size:(idx+1)*config.test_batch_size]
            batch = [ np.load(batch_file) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
#             print ("[*] test data for anomaly detection is loaded")
            for epoch in range(config.test_epoch):
                feed_dict = {self.test_inputs: batch_images} 
                #_, ano_score, res_loss = self.sess.run([z_optim, self.anomaly_score, self.res_loss], feed_dict = feed_dict)
                #ssim_score = self.sess.run([self.ssim_score], feed_dict = feed_dict)
                #print(ssim_score)
                _, ano_score, res_loss, z_vector, dis_f_z, dis_f_input, dis_loss = self.sess.run([z_optim, self.anomaly_score, self.res_loss, 
                                                                  self.ano_z, self.dis_f_z, self.dis_f_input, self.dis_loss], feed_dict = feed_dict)
                #if np.mod(epoch, 1000) == 0:
                #    print("Epoch: [{:02d}], anomaly score: {:.8f}, res loss: {:.8f}" \
                #          .format(epoch, ano_score/self.test_batch_size, res_loss/self.test_batch_size))

#             print("***** Epoch finished: Mean anomaly score: {:.8f}, res loss: {:.8f}, dis loss:{:.8f}" \
#                   .format(np.mean(ano_score), np.mean(res_loss), np.mean(dis_loss)))
            
            # Evaluation
            samples = self.sess.run(self.ano_G)
            errors = batch_images - samples
            errors = np.abs(np.mean(errors, axis = (1,2,3)))
            #print("Test batch: [{:03d}], Pixel Error : {:.8f}".format(idx, np.mean(errors)))
            
            #SSIM (Structrual Similarity measure)
            #ssim_score = self.ssim_evaluate_in_batch(samples, batch_images)
            #print("Test batch: [{:03d}], SSIM score : {:.8f}".format(idx, np.mean(ssim_score)))
            
            # Gram matrix score h2
            gram_matrix_score_h2 = self.sess.run(self.gram_score_h2, feed_dict = feed_dict)
            
            # Gram matrix score
            gram_matrix_score = self.sess.run(self.gram_score, feed_dict = feed_dict)
            #print(gram_matrix_score.shape)
            #print("Test batch: [{:03d}], Gram matrix score : {:.8f}".format(idx, np.mean(gram_matrix_score)))
            
                
            if config.ano_result_save:
                _path = './test_data/' + self.dataset_name + '/'
                #path = os.path.join(_path, config.test_result_dir)
                if not os.path.isdir(_path):
                    os.mkdir(_path)
                filename = test_data_list[idx].split('/')[-1].split('.')[0]
                #np.save(os.path.join(_path, filename)+'_sample',samples)
                #np.save(os.path.join(_path, filename)+'_error',errors)
                #np.save(os.path.join(_path, filename)+'_zvector',z_vector)
                #np.save(os.path.join(_path, filename)+'_dis_fz_vector',dis_f_z)
                #np.save(os.path.join(_path, filename)+'_dis_fi_vector',dis_f_input)
                    
                if not os.path.isdir(_path + 'test_perf'):
                    os.mkdir(_path + 'test_perf')
                
                if subset_idx is None:
                    #csv_name = "./test_data/test_perf/ano_result.csv"
                    csv_name = _path + 'test_perf/ano_result.csv'
                    
                if os.path.exists(csv_name):
                    with open(csv_name, 'a') as fd:
                        writer = csv.writer(fd)
                        for batch_idx in range(0,self.test_batch_size):
                            writer.writerow([batch_files[batch_idx], ano_score[batch_idx], res_loss[batch_idx], dis_loss[batch_idx], gram_matrix_score_h2[batch_idx], gram_matrix_score[batch_idx]])
#                             writer.writerow([batch_files[batch_idx], ano_score[batch_idx], res_loss[batch_idx], dis_loss[batch_idx], errors[batch_idx]])
#                             writer.writerow([batch_files[batch_idx], ano_score[batch_idx], 
#                                             gram_matrix_score_h2[batch_idx], gram_matrix_score[batch_idx], errors[batch_idx]])
                            #writer.writerow([batch_files[batch_idx], ano_score[batch_idx], ssim_score[batch_idx], errors[batch_idx]])
                            
                else:
                    with open(csv_name, 'w') as fd:
                        writer = csv.writer(fd)
                        for batch_idx in range(0,self.test_batch_size):
                            writer.writerow([batch_files[batch_idx], ano_score[batch_idx], res_loss[batch_idx], dis_loss[batch_idx], gram_matrix_score_h2[batch_idx], gram_matrix_score[batch_idx]])
                            #writer.writerow([batch_files[batch_idx], ano_score[batch_idx], res_loss[batch_idx], dis_loss[batch_idx], errors[batch_idx]])
#                             writer.writerow([batch_files[batch_idx], ano_score[batch_idx], 
#                                             gram_matrix_score_h2[batch_idx], gram_matrix_score[batch_idx], errors[batch_idx]])
                            #writer.writerow([batch_files[batch_idx], ano_score[batch_idx], ssim_score[batch_idx], errors[batch_idx]])
                            
        #end_t = time.time()
        #print('[Info] Processing time for a batch :' + str(end_t-start_t))
                    

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)
      
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        if self.saved_file_name:
            ckpt_name = os.path.join(checkpoint_dir, self.model_dir, self.saved_file_name)
            self.saver.restore(self.sess, ckpt_name)
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter
            else:
                print(" [*] Failed to find a checkpoint")
                return False, 0
