import numpy
import cv2
import pickle
import scipy
#这么详细的注释，不得感谢下我，正常情况下我一般写代码不注释
def read(file):
    f=open(file,'r')
    str=f.readlines()
    return str
#激活函数，模拟生物神经元的应激反应，当达到某个阈值才会产生电反应，而不是对输入的数据做线性处理
def expit(data):
    return scipy.special.expit(data)

class NnetWork:
    #初始化函数
    #创建三层神经网络，理论上两层也可以用，只是深度不深，可能导致这个神经网络不太聪明，滑稽滑稽
    def __init__(self,n1_num,n2_num,n3_num):
        #输入层节点个数
        self.n1_num=n1_num
        #隐藏层节点个数
        self.n2_num=n2_num
        #输出层节点个数
        self.n3_num=n3_num
        #学习率，默认0.3
        self.learnRate=0.3
        #创建一个输入到隐藏的权重矩阵
        self.w1=numpy.random.normal(0.0, pow(self.n1_num, -0.5),(self.n2_num, self.n1_num))

      
        #创建一个隐藏到输出的权重矩阵
        self.w2=numpy.random.normal(0.0, pow(self.n2_num, -0.5),(self.n3_num, self.n2_num))
        pass
    #训练函数，更新权重值
    #训练的本质是计算权重，权重可以理解为程序学到的记忆
    def xun(self,inputsL,targetsL):
        inputs=numpy.array(inputsL,ndmin=2).T
        in_n2=numpy.dot(self.w1, inputs)

        ou_n2=expit(in_n2)

        in_n3=numpy.dot(self.w2,ou_n2)
        ou_n3=expit(in_n3)


        targets=numpy.array(targetsL,ndmin=2).T
        #计算误差
        e3=targets-ou_n3
        e2=numpy.dot(self.w2.T,e3)
        #反向更新权重，根据梯度演算公式，这个地方我不是很懂
        #核心公式，神经网络进行学习记忆
        self.w2 +=self.learnRate * numpy.dot((e3 *ou_n3 * (1.0- ou_n3)),numpy.transpose(ou_n2))
        self.w1 +=self.learnRate * numpy.dot((e2 *ou_n2 * (1.0- ou_n2)),numpy.transpose(inputs))
        pass
    #使用这个模型
    def use(self,inputsL):
        #第一层和权重w1计算隐藏层输入，隐藏层神经元节点使用激活函数模拟生物神经元处理信号，计算得到隐藏层输出，然后与权重w2计算得到输出层输入数据，然后与输出层运算输出给输出层神经元
        inputs=numpy.array(inputsL,ndmin=2).T
        in_n2=numpy.dot(self.w1, inputs)
        ou_n2=expit(in_n2)
        in_n3=numpy.dot(self.w2,ou_n2)
        ou_n3=expit(in_n3)
        return ou_n3
        pass
def xun():
    n=NnetWork(784,100,10)
    i=0
    xs_list=read("C:\\Users\\hua'wei\\Desktop\\mnist_train.csv")
    for xs in xs_list:
        xl=xs.strip().split(",")
        flag=xl[0]
        data=(numpy.asfarray(xl[1:])/255.0*0.99)+0.01
        t=numpy.zeros(n.n3_num)+0.01
        t[int(flag)]=0.99
        print("训练数据:",flag,len(data))
        n.xun(data,t)
        if i>=10000:
            break
        i=i+1
    #保存记忆
    numpy.savetxt("w1.csv",n.w1)
    numpy.savetxt("w2.csv",n.w2)
n=NnetWork(784,100,10)
#加载神经网络记忆
n.w1=numpy.loadtxt("w1.csv")
n.w2=numpy.loadtxt("w2.csv")


xs_list=read("C:\\Users\\hua'wei\\Desktop\\d.csv")
for i in xs_list:
    xl=i.strip().split(",")
    data=(numpy.asfarray(xl[1:])/255.0*0.99)+0.01
    res=n.use(data)
    max=0
    ii=0
    xb=0
    for i in res:
        if i[0]>max:
            max=i[0]
            xb=ii
        ii=ii+1
    img=numpy.asfarray(xl[1:]).reshape((28,28))
    cv2.imshow(str(xb),img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    print("神经网络结果:",xb)