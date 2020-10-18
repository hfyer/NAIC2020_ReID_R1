
#  说明文档  
重识别的行人会梦见摄像头吗 


1.如何复现B榜结果  

&nbsp;1）运行环境及配置   
&nbsp;&nbsp;  本次竞赛使用的服务器配置  
&nbsp;&nbsp;&nbsp;  - 4卡1080Ti  
&nbsp;&nbsp;&nbsp;  - 2卡Titan  
&nbsp;&nbsp;&nbsp;  - CUDA  10.2   
&nbsp;&nbsp;&nbsp;  - pytorch 1.6.0  
&nbsp;&nbsp; 根据fast-reid-master/docs/INSTALL.md 进行环境配置    
&nbsp;&nbsp; 将本项目copy至服务器，在fast-reid-master文件夹内将logs压缩文件解压  
&nbsp;&nbsp; https://pan.baidu.com/s/1JkyFZZ0TrI1rMRU_PAOyEg 提取码：4bmh   

&nbsp;2）数据集存放   
&nbsp;&nbsp; 在fast-reid-master内新建datasets文件夹,其中文件结构如下：    
naic2019  
&nbsp;&nbsp;&nbsp;&nbsp; --round1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --train  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --train_list.txt  
&nbsp;&nbsp;&nbsp;&nbsp;--round2  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--train  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--train_list.txt  
naic2020_round1  
    &nbsp;&nbsp;&nbsp;&nbsp;--image_A  
        &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;--gallery  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--query  
    &nbsp;&nbsp;&nbsp;&nbsp;--train  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--images  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--label.txt  
        
        
&nbsp;3）复现过程  
  &nbsp;&nbsp;该结果由5个模型集成而成，其中模型均在logs/NAIC_All/A中，集成的5个模型其文件夹名分别是：0_269x，0_269x_augmix，0_269x_rcs_augmix，1_101x_rcs， 2_200x_rcs  
  复现过程为：首先对于5个模型，各自计算dist.npy文件，最后运行fastreid下的ensemble_dist.py进行集成，获得最终的R1_B.json提交文件  

分别测试五个模型：需要适当更改configs/PCL下的yml文件中的OUTPUT_DIR设置来保存文件  

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S101.yml --eval-only --num-gpus 4  MODEL.WEIGHTS logs/NAIC_All/A/1_101x_rcs/1_101x_rcs/101x_rcs_model.pth    

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S200.yml --eval-only --num-gpus 4  MODEL.WEIGHTS logs/NAIC_All/A/2_200x_rcs/1_200x_rcs/200x_rcs_model.pth      

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S269.yml --eval-only --num-gpus 4  MODEL.WEIGHTS logs/NAIC_All/A/0_269x/269x_model.pth   

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S269.yml --eval-only --num-gpus 4  MODEL.WEIGHTS logs/NAIC_All/A/0_269x_augmix/269x_augmix_model.pth  

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S269.yml --eval-only --num-gpus 4  MODEL.WEIGHTS logs/NAIC_All/A/0_269x_rcs_augmix/0_269x_rcs_augmix/269x_rcs_augmix_model.pth  

最后在fastreid 中运行 python ensemble_dist.py   
需要适当更改 ensemble_dist.py 中的query_path，gallery_path与dist1_path等路径。  

2.如何训练和测试  

&nbsp;1）A榜如何训练和测试 

A榜训练：首先更改数据集位置， fastreid/data/bulid.py 中根据提示改为 ./datasets/  
其次 configs/PCL中的yml DATASETS项 根据提示改为NAIC_All并适当更改OUTPUT_DIR保存输出日志  

执行 CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S50.yml --num-gpus 4   

进行训练，其中S50.yml可改为S101.yml，S200.yml或S269.yml  

训练时269x的网络使用augmix则设置DO_AUGMIX: True，否则DO_AUGMIX: False  
训练时默认rcj数据增强不开启，需要在fastreid/data/transforms/bulid.py中 找到rcj的注释，取消注释  

训练完成后会自行测试，如果需要另外测试，可以执行：

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S101.yml --eval-only --num-gpus 4  MODEL.WEIGHTS logs/NAIC_All/A/1_101x_rcs/1_101x_rcs/101x_rcs_model.pth  

其中 MODEL.WEIGHTS 以及--config-file位置可适当自行更改

&nbsp;2）B榜如何测试  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先更改数据集位置， fastreid/data/bulid.py 中根据提示改为 "/home/zhangzhengjie/workspace/image_B" 可根据image_B存储的位置，自行适当修改  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其次 configs/PCL中的yml DATASETS项 根据提示改为NAIC_test并适当更改OUTPUT_DIR保存输出日志  

执行 CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S101.yml --eval-only --num-gpus 4  MODEL.WEIGHTS logs/NAIC_All/A/1_101x_rcs/1_101x_rcs/101x_rcs_model.pth 

其中MODEL.WEIGHTS 以及--config-file位置可适当自行更改

3.项目介绍  
&nbsp;1）模型配置位置  
&nbsp;&nbsp;&nbsp;&nbsp;  configs/PCL下的yml文件，具体模型配置参数可以在yml文件中查看  
&nbsp;2）模型介绍  
网络结构：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用 resnest 作为backbone 并且加入IBN层，在BN上由于多卡的原因使用了syncBN
最后的FC分类层使用了 circleSoftmax层替换，同时Pooling 方式由传统的avg pooling改成了可学习的gempool的方式

Loss：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;不带labelSmooth的CrossEntropyLoss  hard TripletLoss  

测试：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用了AQE和 rerank 为了加快测试速度将batchsize设置为 512 

优化器Adam 使用了WarmupCosineAnnealingLR

训练时将图片由256 * 128拉大至384 * 192 batchsize 64 

数据增强方面使用了fastreid框架自带的augmix ，以及发现的trick 0.5概率三通道随机交换（记为rcs），

同时初赛最终的方案为： 上述模型的101层版本(rcs)， 200层版本(rcs)， 269x版本， 以及269层版本（rcs），和269层版本(rcs,augmix)，总共五个模型的集成版本。 

&nbsp;3）trick  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;数据增强
使用了三通道0.5概率随机交换的数据增强trick  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;具体代码位置  fastreid/data/transforms/transforms.py中的RandomShuffleChannel类
