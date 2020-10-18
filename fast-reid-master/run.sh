
#测试
#CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S101.yml --eval-only --num-gpus 4  MODEL.WEIGHTS logs/NAIC_All/new/1_101x/model_final.pth 

#训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file ./configs/PCL/S269.yml --num-gpus 4



#Round 1 final model 多模集成 
# 101x(60.93) 
# rcs 101x(63.03)

# 200x(61.35) 
# rcs 200x(62.41) 

# 269x(62.27) 
# 269x_AUGMIX(62.33)

# rcs 269x(63.03)
# rcs 269x augmix(62.29)




#A榜
#rcs 101x(63.03) rcs 200x(62.41)   269x(62.27)                                          en (64.99) 
#rcs 101x(63.03) rcs 200x(62.41)  269x_augmix(62.33)                                    en (65.35)
#rcs 101x(63.03) rcs 200x(62.41)  269x_augmix(62.33) 269x(62.27) rcs_269x_augmix(62.29) en(65.86)


# R1 final model
#B榜
# rcs101x rcs200x 269xaugmix  269x rcs269xaugmix   en(60.31)
#101x 200x rcs101x rcs200x 269xaugmix  269x rcs269xaugmix  en(59.98)