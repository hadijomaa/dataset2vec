import tensorflow as tf

class D2V(object):
    def __init__(self,config,seed=0):
        self.config = config
        tf.set_random_seed(seed)
        # Define attributes
        self.X = None
        self.Y = None
        self.N_features  = None
        self.N_classes   = None
        self.N_instances = None
        self.N_pos       = None
        self.Y_hat          = None
        self.pred_variables = None
        self.prediction_model_update = None
        self.create_model()

    def create_model(self):
        self.define_placholders()
        self.create_prediction_model()
        self.create_loss()
        self.create_update_rules()
        
    def define_placholders(self):
        with tf.variable_scope('Placeholders'):
            self.X = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='Dataset')
            self.Y = tf.placeholder(dtype=tf.float32,shape=(None,1),name='labels')
            self.N_features     = tf.placeholder(dtype=tf.int32, shape=(2*self.config['batch_size']), name='N_features')
            self.N_classes      = tf.placeholder(dtype=tf.int32, shape=(2*self.config['batch_size']), name='N_classes')
            self.N_instances    = tf.placeholder(dtype=tf.int32, shape=(2*self.config['batch_size']), name='N_instances')
            self.N_pos        = tf.placeholder(dtype=tf.int32, shape=(), name='num_pos')
            
    def residual_block(self,x,nhidden,activation=tf.nn.relu,prefix='Resblock'):
            layer = x + 0
            for idx, units in enumerate(nhidden):
                layer = tf.layers.dense(inputs=layer,
                        activation=None,
                        units=units,
                        name=prefix+str(idx))
                if idx < (len(nhidden) - 1):
                    layer = activation(layer)
            return activation(layer + x)

    def prepool_layers(self):
        layer = self.X
        for idx, desc in enumerate(self.config['prepool_layers']):
            if desc['type'] == 'dense':
                layer = tf.layers.dense(inputs=layer,
                                    activation=desc['activation'],
                                    units=desc['units'],
                                    name='PreDense'+str(idx))
                units = desc['units']
            elif desc['type'] == 'res':
                layer = self.residual_block(layer,nhidden=desc['hidden'],activation=desc['activation'],prefix='PreResBlock')
                units = desc['hidden'][-1]
                
        size_splits = tf.multiply(self.N_classes,
                                  tf.multiply(self.N_features,
                                              self.N_instances,
                                              name='N_features_times_N_instances'),
                                  name='N_classes_times_N')
        layer       = tf.split(layer,num_or_size_splits=size_splits,axis=0,name='dataset_split_1')                
        layer_list  = []
        for i,batchX in enumerate(layer):
            temporary_layer     = tf.reshape(batchX,shape=(-1,self.N_classes[i],self.N_features[i],self.N_instances[i],units))
            temporary_layer     = tf.reduce_mean(temporary_layer,axis=3)
            layer_list.append(tf.reshape(temporary_layer,shape=(-1,units)))
        layer = tf.concat(layer_list,axis=0,name='concat_layers_1')
        return layer
    
    def prediction_layers(self,layer):
        for idx, desc in enumerate(self.config['prediction_layers']):
            layer = tf.layers.dense(inputs=layer,
                                    activation=desc['activation'],
                                    units=desc['units'],
                                    name='Dense'+str(idx))
            units = desc['units']
        size_splits_2 = tf.multiply(self.N_features,self.N_classes,name='N_features_times_N_classes')
        layer = tf.split(layer,num_or_size_splits=size_splits_2,name='dataset_split_2')
        layer_list = []
        for i,batchX in enumerate(layer):
            temporary_layer = tf.reshape(batchX,shape=(-1,self.N_features[i]*self.N_classes[i],units))
            layer_list.append(tf.reduce_mean(temporary_layer,axis=1))
        layer = tf.concat(layer_list,axis=0,name='concat_layers_2')
        return layer

    def postpool_layers(self,layer):
        for idx, desc in enumerate(self.config['postpool_layers']):
            if desc['type'] == 'dense':
                layer = tf.layers.dense(inputs=layer,
                                    activation=desc['activation'],
                                    units=desc['units'],
                                    name='PostDense'+str(idx))
            elif desc['type'] == 'res':
                layer = self.residual_block(layer,nhidden=desc['hidden'],activation=desc['activation'],prefix='PostResBlock')
        return layer
    
    def create_prediction_model(self):
        with tf.variable_scope('PredictionNetwork'):
            layer = self.prepool_layers()
            layer = self.prediction_layers(layer)
            layer = self.postpool_layers(layer)
            layer = tf.layers.dense(inputs=layer,
                                    activation=None,
                                    units=self.config['c_units'],
                                    name='FinalDense')
            self.phi = layer
            self.c_units = self.config['c_units']
            self.Y_hat = layer
            self.pred_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='PredictionNetwork')

    def euclidean_distance(self,Y_hat=None):
        if Y_hat is None:
            Y_hat = self.Y_hat
        layer   = tf.reshape(Y_hat,shape=[-1,2,self.c_units],name='pair_batches')
        layer   = tf.subtract(layer[:,0],layer[:,1],name='vector_difference')
        layer   = tf.square(layer,name='difference_squared')
        layer   = tf.reduce_sum(layer,axis=-1,name='euclidean_distance')
        return layer
    
    def get_logits(self):
        layer = self.euclidean_distance()
        return tf.exp(- layer)
        
    
    def create_loss(self):

        with tf.variable_scope('LossNetwork'):
            layer       = self.euclidean_distance()
            p_loss_per  = tf.log(tf.exp(- layer[:self.N_pos]) + self.config['eps'], name='positive_pairs_loss')
            n_loss_per  = tf.log(1 - tf.exp(- layer[self.N_pos:]) + self.config['eps'], name='negative_pairs_loss')
            self.pos_loss    = - tf.reduce_mean(p_loss_per)
            self.neg_loss    = - tf.reduce_mean(n_loss_per)
            self.L           = self.pos_loss + self.config['neg_loss_weight']*self.neg_loss

    def create_update_rules(self):
        with tf.variable_scope('UpdateRules'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='PredictionNetwork')
            with tf.control_dependencies(update_ops):
                unclipped_grads = tf.gradients(self.L, self.pred_variables)
                if self.config['max_grad_norm'] is not None:
                    clipped_grads, _ = tf.clip_by_global_norm(unclipped_grads, self.config['max_grad_norm'])
                else:
                    clipped_grads = unclipped_grads
                self.pred_unclipped_grad_norm = tf.global_norm(unclipped_grads)
                self.pred_clipped_grad_norm = tf.global_norm(clipped_grads)
                self.prediction_model_update = tf.train.AdamOptimizer(self.config["eta_pred"]).\
                    apply_gradients(zip(clipped_grads, self.pred_variables))
            
    def update_prediction_model(self, sess, X, N_features,N_classes,N_instances,N_pos):
            _, loss,pos_loss,neg_loss, grad_norm_unclipped, grad_norm_clipped = \
                sess.run(fetches=[self.prediction_model_update, self.L,self.pos_loss,self.neg_loss,
                                  self.pred_unclipped_grad_norm, self.pred_clipped_grad_norm],
                         feed_dict={self.X: X,
                                    self.N_features: N_features,
                                    self.N_classes:  N_classes,
                                    self.N_instances: N_instances,
                                    self.N_pos: N_pos})
            return loss,pos_loss,neg_loss, grad_norm_unclipped, grad_norm_clipped