import tensorflow as tf
# 验证码生成库
from captcha.image import ImageCaptcha
import string
import random
# 验证码图像处理库
from PIL import Image
import numpy as np
import cv2


# 其他
import os
import copy
import time

CAPTCHA_LEN = 4               # 验证码长度
CAPTCHA_LIBSIZE = 63          # 验证码字符库大小 = 数字 + 大写字母 + 小写字母  + 占位符'_' = 10+26+26+1 = 63
IMG_WIDTH = 160               # 图像宽
IMG_HEIGHT = 60               # 图像高
ACCURANCY = 0.99               # 预期训练正确率
ROOT_DIR = [r'./train',r'./test',r'./model'] # r'./model_612'
CHILD_DIR = [r'/srcimg',r'/blackimg',r'/proimg']
TXT_DIR = r'./txtfile'
GPU_INDEX = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

class Captcha:
    def __init__(self,mode,save_img = False):
        self.saveflag = save_img
        # 根据不同模式决定保存的路径
        self.srcpath = ROOT_DIR[mode] + CHILD_DIR[0]
        self.blackpath = ROOT_DIR[mode] + CHILD_DIR[1]
        self.propath = ROOT_DIR[mode] + CHILD_DIR[2]
    
    '''随机生成指定长度的文本''' 
    def generate_text(self,length = CAPTCHA_LEN):
        # 产生字母元组
        letter = string.ascii_letters
        # 产生数字元组
        number = string.digits
        # 随机生成四个字符，组装成字符串
        text = ''.join(random.sample(letter+number,length))
        return text

    '''随机生成验证码图片, 格式PIL, 默认尺寸 60*160'''
    def generate_captcha(self,length = CAPTCHA_LEN):
        image = ImageCaptcha()
        # 随机生成验证码文本
        captcha_text = self.generate_text(length)
        # 生成验证码图像,PIL格式
        captcha = image.generate(captcha_text)
        captcha_image = Image.open(captcha)
        # 图像预处理，去掉噪点和干扰横线
        captcha_image,img_group = self.Image_processing_pro(captcha_image,captcha_text)
        if captcha_image is None:
            raise ValueError("Captcha_Image is None")
        return captcha_text, captcha_image,img_group

    '''二维码图像预处理'''
    def Image_processing_pro(self,image_PIL,name):
        # 自建腐蚀核滤除孤立白色区域
        def erode_img(black_img):
            # 腐蚀核
            kernel =  np.array((
                    [1,1,1,1,1,1],
                    [1,0,0,0,0,1],
                    [1,0,0,0,0,1],
                    [1,0,0,0,0,1],
                    [1,0,0,0,0,1],
                    [1,1,1,1,1,1]
                    ),dtype='uint8')
            # 添加宽度为1的padding
            image_padd = cv2.copyMakeBorder(black_img,1,1,1,1, cv2.BORDER_CONSTANT, value=0)
            # 深拷贝
            image_filted = copy.deepcopy(image_padd)
            # 滤波器移动步长
            stride = 1
            # 腐蚀滤波消去孤立点
            for col in range(0,image_padd.shape[0]-kernel.shape[0]+1,stride):
                for row in range(0,image_padd.shape[1]-kernel.shape[1]+1,stride):
                    # 取原图与滤波器同等大小的区域
                    roi = image_padd[col:col+kernel.shape[1],row:row+kernel.shape[0]]
                    # 对应项相乘
                    dot = roi*kernel
                    # 乘积结果为0时,将该区域所有像素值置0，否则保留原图像
                    if not dot.any():
                        image_filted[col:col+kernel.shape[1],row:row+kernel.shape[0]] = dot
                        row += kernel.shape[1]  
            # 去掉外边框                 
            result = image_filted[1:image_filted.shape[0]-1,1:image_filted.shape[1]-1]
            return result
        # PIL转Opencv格式
        image = cv2.cvtColor(np.asarray(image_PIL),cv2.COLOR_RGB2BGR)
        # 灰度化
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Otsu 二值化，文本为白色，背景为黑色
        # _,image_black = cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        image_black=cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,17,3)
        img_erode = erode_img(image_black)
        #     # 定义一个3*3的卷积核
        #     kernel=np.ones((3,3),np.uint8)
        #     # 开运算：先腐蚀后膨胀
        #     open_img=cv2.morphologyEx(image_black,cv2.MORPH_OPEN,kernel)
        #     # kernel=np.ones((4,4),np.uint8)
        #     # erosion = cv2.erode(image_black,kernel,iterations = 1)
        #     # kernel=np.ones((3,3),np.uint8)
        #     # open_img = cv2.dilate(erosion,kernel,iterations = 1)

        image_group = [image,image_black,img_erode]
        
        # Opencv格式转PIL，再转为array
        captcha_img = np.array(Image.fromarray(img_erode))
        #captcha_img = np.array(Image.fromarray(image_black))
        return captcha_img,image_group

    '''二维码图像预处理'''
    def Image_processing(self,image_PIL,name):
        # 自建腐蚀核滤除孤立白色区域
        def erode_img(black_img):
            # 腐蚀核
            kernel =  np.array(([1,1,1,1,1,1],
                    [1,0,0,0,0,1],
                    [1,0,0,0,0,1],
                    [1,0,0,0,0,1],
                    [1,0,0,0,0,1],
                    [1,1,1,1,1,1]),dtype='uint8')
            # 添加宽度为1的padding
            image_padd = cv2.copyMakeBorder(black_img,1,1,1,1, cv2.BORDER_CONSTANT, value=0)
            # 深拷贝
            image_filted = copy.deepcopy(image_padd)
            # 滤波器移动步长
            stride = 1
            # 腐蚀滤波消去孤立点
            for col in range(0,image_padd.shape[0]-kernel.shape[0]+1,stride):
                for row in range(0,image_padd.shape[1]-kernel.shape[1]+1,stride):
                    # 取原图与滤波器同等大小的区域
                    roi = image_padd[col:col+kernel.shape[1],row:row+kernel.shape[0]]
                    # 对应项相乘
                    dot = roi*kernel
                    # 乘积结果为0时,将该区域所有像素值置0，否则保留原图像
                    if not dot.any():
                        image_filted[col:col+kernel.shape[1],row:row+kernel.shape[0]] = dot
                        row += kernel.shape[1]  
            # 去掉外边框                 
            result = image_filted[1:image_filted.shape[0]-1,1:image_filted.shape[1]-1]
            return result
        # PIL转Opencv格式
        image = cv2.cvtColor(np.asarray(image_PIL),cv2.COLOR_RGB2BGR)
        # 灰度化
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Otsu 二值化，文本为白色，背景为黑色
        # _,image_black = cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        image_black=cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,17,3)
        # img_erode = erode_img(image_black)
        #     # 定义一个3*3的卷积核
        #     kernel=np.ones((3,3),np.uint8)
        #     # 开运算：先腐蚀后膨胀
        #     open_img=cv2.morphologyEx(image_black,cv2.MORPH_OPEN,kernel)
        #     # kernel=np.ones((4,4),np.uint8)
        #     # erosion = cv2.erode(image_black,kernel,iterations = 1)
        #     # kernel=np.ones((3,3),np.uint8)
        #     # open_img = cv2.dilate(erosion,kernel,iterations = 1)

        image_group = [image,image_black,None]
        
        # Opencv格式转PIL，再转为array
        captcha_img = np.array(Image.fromarray(image_black))
        return captcha_img,image_group

    def save_img2file(self,img_group,imgname_group,batch_text_group):
        if self.saveflag:
            print("正在保存图片...")
            name_number = 0 # 本地有同名文件时的后缀
            label_number = 0 # 本地有同名文件时的后缀
            for imgs,name,label in zip(img_group,imgname_group,batch_text_group):
                if(name != label):
                    name_tmp = name
                    label_tmp = label
                    # 重名处理,增加后缀: '_1','_2',...
                    while(os.path.isfile(os.path.join(self.srcpath,label_tmp + '.jpg'))):
                        label_number +=1
                        label_tmp = label_tmp + '_' + str(label_number) 
                    while(os.path.isfile(os.path.join(self.blackpath,name_tmp + '.jpg'))):
                        name_number +=1
                        name_tmp = name + '_' + str(name_number) 
                    if label_number != 0:
                        label += '_' + str(label_number)
                    if name_number != 0:
                        name += '_' + str(name_number)
                    # 分文件夹保存图片
                    cv2.imwrite(os.path.join(self.srcpath,label + '.jpg'),imgs[0])
                    cv2.imwrite(os.path.join(self.blackpath,name + '.jpg'),imgs[1])
                    cv2.imwrite(os.path.join(self.propath,name + '.jpg'),imgs[2])
            print("成功保存图片")
        
    '''text转向量'''
    # 向量长度 = 字符长度 * 字符库大小 = 4 * 63 = 252
    def text2vec(self,text):
        if len(text) > CAPTCHA_LEN:
            raise ValueError('验证码最长%d个字符'%CAPTCHA_LEN)
        # 初始化向量
        vector = np.zeros(CAPTCHA_LEN * CAPTCHA_LIBSIZE,dtype=np.int32)
        # 利用 ASCII 码获取字符对应的编号位置, 顺序为 数字->大写字母->小写字母，
        # 0-9 ∈ [48,57] A-Z ∈ [65,90] a-z ∈ [97,122]
        def get_index(c):
            index = CAPTCHA_LIBSIZE - 1 # 默认为占位符
            ascii = ord(c)
            # 下划线
            if c == '_': 
                return index
            index = ascii - 48 # 从数字0算起
            # 字母
            if index > 9:
                index = ascii - 55
                # 不是大写字母
                if ascii > 90:
                    # 非法字符
                    if ascii < 97:
                        raise ValueError('非法字符！')
                    # 小写字母
                    else:
                        index = ascii - 61
            return index
        # 将字符对应位置的vector的值置1
        for i, char in enumerate(text):
            index = i * CAPTCHA_LIBSIZE + get_index(char)
            vector[index] = 1
        return vector

    '''向量转回text'''
    def vec2text(self,vector):
        # 注意这里的vector是已经得到index的,即vetor= [[1, 99, 136, 191],...],期望结果text = ['1aA2',...]
        def idx2text(indexs):
            if len(indexs) > CAPTCHA_LEN:
                raise ValueError('验证码最长%d个字符'%CAPTCHA_LEN)
            text = ''
            for index in indexs:
                char_idx = index % CAPTCHA_LIBSIZE
                if char_idx < 10:
                    text += chr(char_idx + ord('0'))
                elif char_idx < 36:
                    text += chr(char_idx - 10 + ord('A'))
                elif char_idx < 62:
                    text += chr(char_idx - 36 + ord('a'))
                else:
                    text += '_'
            return text
        text_list = list(map(idx2text,vector))
        return text_list

class Captcha_CNN:
    def __init__(self,mode = 0,save_img = False):
       # 实例化验证码对象
       self.captcha = Captcha(mode,save_img)
       if not self.envir_test():
            raise ValueError("初始化失败")

       # 定义数据集的占位符 行不定表示任意数量,列为60*160表示将图像展平成向量
       self.X = tf.placeholder(tf.float32, [None, IMG_HEIGHT*IMG_WIDTH],name='X')
       # 定义真实标签值的占位符 行不定表示任意数量,列为4*63表示向量化的文本
       self.Y_ = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CAPTCHA_LIBSIZE],name='Y_')
       # 定义防过拟合参数, 以keep_prob的概率决定是否保留元素,不保留则置0,保留则 为原值的 1/keep_prob 倍
       self.keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')

    '''train和test共有变量初始化'''
    def parement_init(self):
       # 获取目标预测结果标签值为1的元素所在位置的编号(因为只有0,1)
       self.predict_idx = tf.argmax(tf.reshape(self.output, [-1, CAPTCHA_LEN, CAPTCHA_LIBSIZE]), 2)    
       # 获取真实标签值为1的元素所在的位置编号   
       self.real_idx = tf.argmax(tf.reshape(self.Y_, [-1, CAPTCHA_LEN, CAPTCHA_LIBSIZE]), 2)
       # 判断检测结果与真实值是否相等,结果为布尔元组[True,Ture,Flase,True]
       correct_pred = tf.equal(self.predict_idx, self.real_idx)    
       # 计算正确率, tf.cast为类型转换,转换结果[1.,1.,0.,1.],则accuracy = 0.75
       self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  

    '''环境测试'''
    def envir_test(self):  
        # 创建模型保存文件夹
        for i,root_dir in enumerate(ROOT_DIR):
            for child_dir in CHILD_DIR:
                dir = root_dir
                if(i<2):
                    dir += child_dir 
                if not os.path.exists(dir):
                    os.makedirs(dir)
                    print('成功创建文件夹:%s'%dir)
        # 创建日志保存文件夹
        if not os.path.exists(TXT_DIR):
            os.makedirs(TXT_DIR)
            print('成功创建日志文件夹')
        # 验证码生成测试
        captch_text, captcha_image,_ = self.captcha.generate_captcha() # 随机产生一个验证码样本
        if len(captch_text) > CAPTCHA_LEN:
            raise ValueError('验证码长度超出%d '%CAPTCHA_LEN)
        elif captcha_image is None:
            raise ValueError("验证码图像生成失败")
        elif captcha_image.shape != (IMG_HEIGHT,IMG_WIDTH):
            raise ValueError("验证码图像尺寸有误")
        else:
            print('环境测试成功!')
            return True
        return False

    '''生成一批指定数量的样本'''
    def get_next_batch(self,batch_size=64):
        # 图像数据
        batch_X = np.zeros([batch_size, IMG_HEIGHT*IMG_WIDTH])
        # 图像标签<向量形式>
        batch_Y_ = np.zeros([batch_size, CAPTCHA_LEN * CAPTCHA_LIBSIZE])
        # 图像标签<text形式>
        batch_text = []
        # 处理图像组<包含 原图、二值化图、滤波图>
        batch_img_group = []
        for i in range(batch_size):
            # 随机产生一个验证码样本
            captch_text, captcha_image, img_group = self.captcha.generate_captcha()
            # 图像拍平
            batch_X[i, :] = captcha_image.flatten() / 255
            # 文本转为编码向量
            batch_Y_[i, :] = self.captcha.text2vec(captch_text)
            # 
            batch_text.append(captch_text)
            batch_img_group.append(img_group)

        return batch_X, batch_Y_, batch_text,batch_img_group

    '''定义卷积神经网络模型'''
    def model(self):
        w_alpha = 0.01 # 权重系数
        b_alpha = 0.1  # 偏重系数
        '''第 0 层: 输入层, 图片大小: 60x160x1'''
        # 重塑张量
        x = tf.reshape(self.X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
        
        '''第 1、2 层: 卷积池化层, 卷积核尺寸: 3x3, 卷积核个数: 32, 输出: 30x80*32'''
        '''卷积运算->激活函数->最大池化->drop防过拟合  https://www.cnblogs.com/skyfsm/p/6790245.html'''
        # 随机生成服从正态分布的卷积核及偏重,乘以系数,使其小一点又不至于为0
            # 随即产生: tf.random_normal(shape)
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32])) 
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))            
        # 卷积层: 执行卷积运算,输出: 60x160x32
            # 定义卷积: tf.nn.conv2d(输入图像,卷积核,步长[batch,stride,stride,channels],'边缘填充')  https://www.jianshu.com/p/c72af2ff5393
            # 添加偏重: tf.nn.bias_add(卷积,偏重) https://blog.csdn.net/mieleizhi0522/article/details/80416668 
        conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
        # 激励层: 用激活函数将卷积运算结果作非线性映射
            # 激活函数Relu: tf.nn.relu(卷积运算结果),大于0保留,小于0置0
        conv1 = tf.nn.relu(conv1)
        # 池化层: 压缩数据,输出: 30x80x32 
            # 最大值池化: tf.nn.max_pool(value,池化窗口大小[batch,height,width,channels],步长[batch,stride,stride,channels],'边缘填充') https://www.jianshu.com/p/1d73fd1a256e
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 防过拟合层: 随机保留
            # tf.nn.dropout(value, dropout概率) https://www.jianshu.com/p/c9f66bc8f96c   
        conv1 = tf.nn.dropout(conv1, self.keep_prob)

        '''第 3、4 层: 卷积池化层, 卷积核尺寸: 3x3, 卷积核个数: 64, 输出: 15x40*64'''
        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        '''第 5、6 层: 卷积池化层, 卷积核尺寸: 3x3, 卷积核个数: 64, 输出: 8x20*64'''
        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)

        ''''第 7 层: 全连接层, 卷积核尺寸: 视作8x20, 卷积核个数: 1024, 输出: 1x1024*1''' 
        '''https://www.cnblogs.com/MrSaver/p/10357334.html'''
        '''可以参考MINST手写数字理解：https://www.cnblogs.com/fydeblog/p/7455233.html '''
        '''特征分类'''
        w_f4 = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
        b_f4 = tf.Variable(b_alpha * tf.random_normal([1024]))
        # 将卷积层输出拍平为 1x(8*20*64 = 10240)
        full_cnc = tf.reshape(conv3, [-1, w_f4.get_shape().as_list()[0]])
        # 10240个输入值与卷积核w_b 进行矩阵乘法再加上偏重,相当于卷积核变成8x20
        full_cnc = tf.nn.relu(tf.add(tf.matmul(full_cnc, w_f4), b_f4))
        # 防过拟合
        full_cnc = tf.nn.dropout(full_cnc, self.keep_prob)

        '''第 8 层: 全连接层,卷积核尺寸: 1x1, 卷积核个数: 等于标签种类数, 输出:各标签的预测值'''
        '''标签分类'''
        w_o5 = tf.Variable(w_alpha * tf.random_normal([1024*1, CAPTCHA_LEN * CAPTCHA_LIBSIZE]))
        b_o5 = tf.Variable(b_alpha * tf.random_normal([CAPTCHA_LEN * CAPTCHA_LIBSIZE]))
        output = tf.add(tf.matmul(full_cnc, w_o5), b_o5)
        return output
    
    '''模型训练过程'''
    def train(self,save_img = False):
        with tf.device('/gpu:0'):
            # 创建CNN模型
            self.output = self.model()
            # train与test共有变量初始化
            self.parement_init()
        # sigmoid交叉熵计算损失值, 计算结果取平均值
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.Y_))     # 计算损失
        # Adam 优化器优化算法更新学习率,区别于传统梯度下降法 https://www.cnblogs.com/guoyaohua/p/8542554.html 
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # minimize包含两个操作:compute_gradients()计算梯度,apply_gradients()更新参数  https://www.jianshu.com/p/72948cce955f 
        train_step = optimizer.minimize(loss)         

        # 实例化'模型保存与恢复'对象
        saver = tf.train.Saver() 

        # 定义文本文件
        Loss_file = open(os.path.join(TXT_DIR,'loss.txt'),'w')
        Acc_file = open(os.path.join(TXT_DIR,'accurancy.txt'),'w')

        # 定义会话
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:  
            # 初始化计算图(Model)中的变量
            sess.run(tf.global_variables_initializer())
            # 加载已经训练的的模型参数
            checkpoint = tf.train.get_checkpoint_state(ROOT_DIR[2])
            # 模型读取结果判断
            if checkpoint and checkpoint.model_checkpoint_path:
                print("成功装载已有模型:", checkpoint.model_checkpoint_path)
                # 参数装载
                saver.restore(sess, checkpoint.model_checkpoint_path)

            # 初始化训练批数
            self.train_times = 0
            accuracy_tmp = 0
            print('开始训练...')
            while True:
                try:
                    # 获取一批样本
                    batch_X, batch_Y_,batch_text,img_group = self.get_next_batch(batch_size=64)
                    # 根据初始化要求尝试保留数据集,命名为真实标签
                    self.captcha.save_img2file(img_group,batch_text,batch_text)
                    # 训练并打印当前损失计算结果,sess.run(fetch,feed) fetch决定了哪些节点是期望被激活的,feed是填充placeholder占位符,只在调用它的op中生效
                    _,loss_tmp = sess.run([train_step,loss], feed_dict={self.X: batch_X, self.Y_: batch_Y_, self.keep_prob: 0.75})
                    print('训练批数:%d  当前Loss: %.5f'%(self.train_times, loss_tmp))
                    Loss_file.write(str(loss_tmp)+'\n')
                    # 每训练100批,进行一次正确率测试
                    if self.train_times and self.train_times % 100  == 0:
                        # 获取100个测试集
                        batch_X_test, batch_Y_test,batch_text,img_group = self.get_next_batch(batch_size=100)
                        # 根据初始化要求尝试保留数据集,命名为真实标签
                        self.captcha.save_img2file(img_group,batch_text,batch_text)
                        # 测试并打印当前正确率
                        accuracy_tmp = sess.run(self.accuracy, feed_dict={self.X: batch_X_test, self.Y_: batch_Y_test, self.keep_prob: 1.0})
                        print('当前正确率:%.4f' % accuracy_tmp)
                        Acc_file.write(str(accuracy_tmp)+'\n')
                        if accuracy_tmp > ACCURANCY or self.train_times >= 30000 :
                            print("正在保存模型...")
                            saver.save(sess, os.path.join(ROOT_DIR[2],str(round(accuracy_tmp,4))), global_step=self.train_times)
                            break
                    self.train_times += 1
                except:
                    if self.train_times > 0:
                        print("强制中断,正在保存模型...")
                        saver.save(sess, os.path.join(ROOT_DIR[2],str(round(accuracy_tmp,4))), global_step=self.train_times - 1)
                        break
        # 释放计算图
        tf.reset_default_graph()
        # 关闭文件
        Loss_file.close()
        Acc_file.close()
        
    '''模型测试过程'''
    def test(self):
        with tf.device('/gpu:0'):
            # 创建CNN模型
            self.output = self.model()
            # train与test共有变量初始化
            self.parement_init()
        # 实例化'模型保存与恢复'对象
        saver = tf.train.Saver()
        print('开始测试...')
        start = time.clock()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            # 获取训练后的模型参数
            checkpoint = tf.train.get_checkpoint_state(ROOT_DIR[2])
            # 模型读取结果判断
            if checkpoint and checkpoint.model_checkpoint_path:
                print("成功导入模型:", checkpoint.model_checkpoint_path)
                # 参数装载
                saver.restore(sess, checkpoint.model_checkpoint_path)
                # 获取100个测试集
                batch_X_test, batch_Y_test,batch_text,img_group= self.get_next_batch(batch_size=1000)
                # 运行模型并返回正确率、预测值结果
                accuracy_tmp,predict_idx_tmp = sess.run((self.accuracy,self.predict_idx), feed_dict={self.X: batch_X_test, self.Y_: batch_Y_test, self.keep_prob: 1.0})
                # 打印正确率
                print("正确率: %.4f"%accuracy_tmp)
                # vector转text
                predict_text = self.captcha.vec2text(predict_idx_tmp)
                # 根据初始化要求尝试保留数据集,命名为预测结果
                self.captcha.save_img2file(img_group,predict_text,batch_text)
            else:
                print("找不到模型")
                return None
        end = time.clock()
        print('耗时: %.4f'%(1000*(end-start)))
        print('平均耗时: %.4f'%(end-start))
        # 释放计算图, 修复Variable不存在Bug  https://blog.csdn.net/HY_JT/article/details/81363634
        tf.reset_default_graph()

if __name__ == "__main__":
    mode = 1 # 0为训练  1为测试
    # 训练模式
    if mode == 0:
        # 实例化对象
        cracker = Captcha_CNN(mode,save_img = False)
        # 开始训练
        cracker.train()
    else:
        # 实例化对象
        cracker = Captcha_CNN(mode,save_img = False)
        # 开始测试
        cracker.test()