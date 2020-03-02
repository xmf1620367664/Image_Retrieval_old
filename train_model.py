# _*_ coding:utf-8 _*_

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from keras.datasets.cifar10 import load_data
from keras.utils import np_utils
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.feature_extraction.text import HashingVectorizer
from keras import layers,models,backend,callbacks
from keras.optimizers import Adam
import argparse

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

class CIFAR():

    def __init__(self,data_path='./data',batch_size=32,input_dt=(32,32,3),
                 classes=10,log_dir='./log/001/',hash_file='hash.csv',
                 feature_size=2048):
        self.input_dt=input_dt
        self.classes=classes
        self.data_path=data_path
        self.batch_size=batch_size
        self.log_dir=log_dir
        self.hash_path=hash_file
        self.fc1=128
        self.feature_size=feature_size
        #self.x=tf.placeholder(tf.float32,shape=[None,self.input_dt])
        self.x=layers.Input(shape=[self.input_dt[0]*self.input_dt[1]*self.input_dt[2]])
        #self.y=tf.placeholder(tf.float32,shape=[None,self.output_dt])
        #self.mnist = input_data.read_data_sets(self.data_path, one_hot=True)
        (self.train_images,self.train_labels),(self.test_images,self.test_labels)=load_data()
        self.train_images,self.test_images=np.reshape(self.train_images,[self.train_images.shape[0],-1]),\
                                           np.reshape(self.test_images,[self.test_images.shape[0],-1])
        self.train_labels,self.test_labels=np_utils.to_categorical(self.train_labels),\
                                           np_utils.to_categorical(self.test_labels)

    def generate_Data(self):
        train_images=self.train_images
        train_labels=self.train_labels
        test_images=self.test_images
        test_labels=self.test_labels
        print('训练集：images={},labels={}'.format(train_images.shape,train_labels.shape))
        print('测试集：images={},labels={}'.format(test_images.shape, test_labels.shape))
        while 1:
            samples,_=self.get_Examples()
            batch_num=samples//self.batch_size
            index=np.random.permutation(samples)
            for i in range(batch_num):
                next_x,next_y=train_images[index[i*self.batch_size:(i+1)*self.batch_size]]\
                             ,train_labels[index[i*self.batch_size:(i+1)*self.batch_size]]
                yield next_x,next_y

    def generate_test_Data(self):
        #mnist = input_data.read_data_sets(self.data_path, one_hot=True)
        while 1:
            _,samples=self.get_Examples()
            batch_num=samples//self.batch_size
            index=np.random.permutation(samples)
            for i in range(batch_num):
                next_x, next_y = self.test_images[index[i*self.batch_size:(i+1)*self.batch_size]],\
                                 self.test_labels[index[i*self.batch_size:(i+1)*self.batch_size]]
                yield next_x, next_y

    def show_Data(self,next_x,show_size=(33,33),correct_rate=0.0):#=(10,10)
        print('show_size',show_size)
        #next_x, _ = self.mnist.train.next_batch(show_size[0]*show_size[1])
        #next_x=next_x[0:show_size[0]*show_size[1]]

        #plt.title('检索准确率:%.4f' % correct_rate)
        f, a = plt.subplots(show_size[0], show_size[1], figsize=(10, 10))
        plt.suptitle('检索准确率:%.4f' % correct_rate)
        #f.suptitle()
        for i in range(show_size[0]):
            #print('i',i)
            for j in range(show_size[1]):
                #print('j',j)
                tmp_x=next_x[i * show_size[0] + j].reshape([32,32,3])
                a[i][j].imshow(tmp_x)
                a[i][j].axis('off')
        plt.show()

    def identity_block(self,input_tensor, kernel_size, filters, stage, block,bn_axis=3):
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def conv_block(self,input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2),
                   bn_axis=3):
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def reshape_Tensor(self,tensor):
        return backend.reshape(tensor,shape=[-1,32,32,3])

    def res_Net50(self,input_tensor,bn_axis=int(3)):
        x=layers.Lambda(self.reshape_Tensor)(input_tensor)
        #x=backend.reshape(input_tensor,shape=[-1,28,28,1])
        #self.x=layers.Input(tensor=x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
        x = layers.Conv2D(64, (3, 3),
                          strides=(1, 1),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')(x)
        x = layers.BatchNormalization( name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(1, 1))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a',strides=(1,1))
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x=layers.Dense(self.fc1,activation='sigmoid',name='fc1')(x)
        x = layers.Dense(self.classes, activation='softmax', name='fc10')(x)
        model=models.Model(self.x,x,name='resnet50')
        return model

    def get_Examples(self):
        return self.train_images.shape[0],\
               self.test_images.shape[0]

    def model_Train(self):

        train_exam,test_exam=self.get_Examples()

        model=self.res_Net50(self.x)
        # 指定回调函数
        logging = callbacks.TensorBoard(log_dir=self.log_dir)
        checkpoint = callbacks.ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_best_only=True, mode='min',
                                     save_weights_only=True, period=1)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        # 指定训练方式
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        if os.path.exists('./' + self.log_dir +'/'+ 'train_weights.h5'):
            model.load_weights('./' + self.log_dir +'/'+ 'train_weights.h5')
        model.fit_generator(self.generate_Data(),
                            steps_per_epoch=max(1,train_exam//self.batch_size),
                            validation_data=self.generate_test_Data(),
                            validation_steps=max(1,test_exam//self.batch_size),
                            epochs=25,
                            initial_epoch=25,
                            callbacks=[logging,checkpoint,reduce_lr,early_stopping])
        model.save_weights(self.log_dir+'/'+'train_weights.h5')
        model.save(self.log_dir+'/'+'train_models.h5')

    def calu_Accuracy(self):
        model = self.res_Net50(self.x)
        if os.path.exists(self.log_dir+'/'+'train_weights.h5'):
            model.load_weights(self.log_dir+'/'+'train_weights.h5')
        else:
            raise RuntimeError('load weights Error!')
        test_images=self.test_images
        test_labels=self.test_labels
        print("正在预测测试集样本")
        result=model.predict(test_images)
        equal=np.equal(np.argmax(test_labels,axis=1),np.argmax(result,axis=1))
        equal=np.where(equal==True,1,0)
        #print(np.cast(equal,np.float32))
        correct_rate=np.mean(equal)
        print("测试集合分类(非检索)准确率为：{}".format(correct_rate))

    def calu_2_Value(self,delta=0.1):
        '''
        train data set
        :return:
        '''
        model = self.res_Net50(self.x)
        if os.path.exists(self.log_dir + '/' + 'train_weights.h5'):
            model.load_weights(self.log_dir + '/' + 'train_weights.h5')
        else:
            raise RuntimeError('load weights Error!')
        get_layer_output=backend.function([model.layers[0].input,backend.learning_phase()],
                                          [model.get_layer('fc1').output])
        train_examples,_=self.get_Examples()
        files=open(self.hash_path,'a+',encoding='utf-8')
        for i in range(train_examples):
            tmp_layer_output=get_layer_output([self.train_images[i]])
            curr_result=tmp_layer_output[0][0]
            len_result=curr_result.shape[0]
            #print(len_result)
            #二值化
            curr_result=np.where(curr_result>delta,1,0)
            tmp_lab=np.argmax(self.train_labels[i])
            #============
            #tmp_pic=self.mnist.train.images[i].reshape([self.input_dt[0]*self.input_dt[1]])
            #============
            #print(tmp_pic)
            files.write(str(tmp_lab)+',')
            for j in range(len_result):
                if j<len_result-1:
                    files.write(str(curr_result[j])+',')
                elif j==len_result-1:
                    files.write(str(curr_result[j]) +'\n')
            #===========
            # for j in range(tmp_pic.shape[0]):
            #     if j<tmp_pic.shape[0]-1:
            #         files.write(str(tmp_pic[j])+',')
            #     elif j==tmp_pic.shape[0]-1:
            #         files.write(str(tmp_pic[j])+'\n')
            #     else:
            #         raise RuntimeError('data desn\'t match')
            #===========
        files.close()

    def calu_Hamming(self,input_feature,data_feature):
        return np.sum(np.abs(input_feature-data_feature))

    def find_Near(self,img,img_label,near_number,delta=0.1):
        files=pd.read_csv(self.hash_path,sep=',',header=None).values
        labels=files[:,0]
        img_label=np.argmax(img_label)
        #files=files.drop(columns=0)
        features=files[:,1:self.feature_size+1]
        #messages=files[:,self.feature_size+1:]
        messages=self.train_images
        #对新传入的img计算哈希值
        model = self.res_Net50(self.x)
        if os.path.exists(self.log_dir + '/' + 'train_weights.h5'):
            model.load_weights(self.log_dir + '/' + 'train_weights.h5')
        else:
            raise RuntimeError('load weights Error!')
        get_layer_output = backend.function([model.layers[0].input, backend.learning_phase()],
                                            [model.get_layer('fc1').output])
        img=img.reshape(self.input_dt[0]*self.input_dt[1]*self.input_dt[2])
        tmp_layer_output = get_layer_output([img])
        curr_result = tmp_layer_output[0][0]
        # 二值化
        curr_result = np.where(curr_result > delta, 1, 0)
        #索引
        hamming_value=[]
        hamming_index=[]
        for i in range(features.shape[0]):
            hamming_value.append(self.calu_Hamming(features[i],curr_result))
        for i in range(near_number*near_number):
            curr_index=hamming_value.index(min(hamming_value))
            hamming_value[curr_index]=max(hamming_value)
            hamming_index.append(curr_index)
        #计算搜索准确率
        labels=np.reshape(np.array(labels),[-1])
        search_labels=labels[hamming_index]
        result=[1 if i==img_label else 0 for i in search_labels]
        result=np.average(result)
        #展示
        #print('传入的图像为：')
        plt.figure()
        plt.title('搜索的图片如下，标签为{}'.format(img_label))
        plt.imshow(img.reshape([self.input_dt[0],self.input_dt[1],self.input_dt[2]]))
        plt.axis('off')
        plt.show()
        self.show_Data(messages[hamming_index],show_size=(near_number,near_number),correct_rate=result)
        print(hamming_value)

        print(files.shape)

if __name__=='__main__':
    #创建一个解析对象
    parser=argparse.ArgumentParser()
    parser.add_argument('--show_size', type=int, default=10)
    parser.add_argument('--test_image_index', type=int, default=3)
    args = parser.parse_args()
    show_size=args.show_size
    test_image_index=args.test_image_index
    with tf.device('/gpu:0'):
        mn=CIFAR()
        #用于数据展示
        mn.show_Data(mn.train_images,show_size=(show_size,show_size))
        #用于模型训练
        #mn.model_Train()
        #计算测试集准确率
        mn.calu_Accuracy()
        #生成特征二值文本
        #mn.calu_2_Value()
        #展示索引图片
        # for near_number in [10,30,50]:
        #     mn.find_Near(mn.test_images[3],mn.test_labels[3],near_number=near_number)
        mn.find_Near(mn.test_images[test_image_index], mn.test_labels[test_image_index], near_number=33)