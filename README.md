# CV——Final project

## Step 1. Process datasets

首先调用learn_mat.py得到对应的训练集，验证集，测试集的label信息。

~~~
python learn_mat.py 
~~~

之后调用数据集处理函数对本次PJ数据集进行处理

~~~shell
python ingre_process-vireo.py 
~~~

## Step 2. Training

训练本模型分为三个步骤，首先需要训练两个预训练网络,通过指定stage表示训练的阶段：

~~~shell
python train.py --dataset 'vireo' --stage 1 --mode 'train' --img_net 'resnet50' --ingre_net 'gru' 
python train.py --dataset 'vireo' --stage 2 --mode 'train' --img_net 'resnet50' --ingre_net 'gru'
~~~

之后将训练好的模型放在指定文件夹下：/VireoFood172/stage1_model和/VireoFood172/stage2_model，作为第三个阶段网络的使用的参数。

最后训练第三个阶段：

~~~shell
python train.py --dataset 'vireo' --stage 3 --mode 'train' --img_net 'resnet50' --ingre_net 'gru'
~~~

## Step 3. Testing

测试：

~~~shell
python test.py --dataset 'vireo' --stage 3 --mode 'test' --img_net 'resnet50' --ingre_net 'gru'
~~~

 
