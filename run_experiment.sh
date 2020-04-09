#　成都
####### pyenv activate python373tf115
host_name=$1
#export gpu=-1


# 23.6
# pyenv activate python36tf150
#export gpu=1


# 房山  全量数据会使得内存不足，运行很慢
# pyenv activate python363tf111
# tensorflow-gpu              1.15.0    版本太高，CUdNN不支持
export gpu=0


export generate_base_model="generate_epoch2_batch189999"
export masker_base_model="epoch0_batch609999_acc_0.634"

# 受到显存限制，batchsize从512降低到256

export config_json=configs/main.json5

#echo "python train.py" ${masker_ranker} "GPU:" ${gpu} "quora_log.txt"
#CUDA_VISIBLE_DEVICES=${gpu} nohup python -u train.py ${config_json}  > quora_log.txt 2>&1 &
#nohup python train.py ${config_json}
#
#python train.py configs/main.json5



#quora
#blocks=2,  enc_layers=2  batch_size=256  vocab=84166
#04/03/2020 12:08:50 > epoch 28 updates 42000 loss: 0.0877 lr: 1.4650e-04 gnorm: 0.3595
#04/03/2020 12:08:50 train loss: 0.1074 lr: 1.4866e-04 gnorm: 0.3694 clip: 0
#04/03/2020 12:08:50 valid loss: 0.3190 auc: 0.9596 f1: 0.8954 acc: 0.8951 [BEST: 0.8968]

#04/03/2020 11:50:26 > epoch 22 updates 32000 loss: 0.2358 lr: 2.4469e-04 gnorm: 0.4625
#04/03/2020 11:50:26 train loss: 0.1761 lr: 2.4469e-04 gnorm: 0.3688 clip: 0
#04/03/2020 11:50:26 valid loss: 0.2881 auc: 0.9550 f1: 0.8926 acc: 0.8889 cost_time: 56.1453 [NEW BEST]

#04/03/2020 10:54:17 Training complete.
#04/03/2020 10:54:17 best dev score 0.8917 at step 36048 (epoch 24).
#04/03/2020 10:54:17 best eval stats [loss: 0.2769 auc: 0.9576 f1: 0.8947 acc: 0.8917 cost_time: 64.2978]

#echo "python train.py" ${masker_ranker} "GPU:" ${gpu} "lcqmc_log.txt"
##CUDA_VISIBLE_DEVICES=${gpu} nohup python -u train.py ${config_json} ${host_name} > lcqmc_log.txt 2>&1 &
#python train.py ${config_json} ${host_name}
##tail -f lcqmc_log.txt

#blocks=1,  enc_layers=2  batch_size=256    'lr': 0.0006, prediction': 'full'    'fusion': 'full',
#04/03/2020 01:39:52 trainable params: 1,183,205
#04/03/2020 01:39:52 trainable params (exclude embeddings): 1,183,205
#average acc=(0.82+0.85+0.84+0.82+0.82+0.85+0.84+0.84+0.82+0.82+0.84)/11=0.83273

#blocks=1,  enc_layers=2  batch_size=256    'lr': 0.0006, prediction':'simple'    'fusion': 'full',
#04/07/2020 10:20:21 trainable params: 1,103,205
#04/07/2020 10:20:21 trainable params (exclude embeddings): 1,103,205
#experiment_times=5/5, test_score_sum=0.833840  稍微好一点点
#04/07/2020 12:24:27 Training time: 0:07:54.   inference_time: 9.2-11.7ms  CPU上纯推理延迟63.6890ms
#[train.py:28]  experiment_times=5/5, ave_test_score=0.824016, max_test_score=0.839600


#blocks=1,  enc_layers=2  batch_size=256    'lr': 0.0006, prediction':'simple  fusion':full  , 换词向量Zero
#04/07/2020 11:20:36 trainable params: 1,103,205
#04/07/2020 11:20:36 trainable params (exclude embeddings): 1,103,205
#04/07/2020 11:28:17 Training time: 0:07:44.
#[train.py:26]  experiment_times=5/5, test_score_sum=0.832352


#blocks=1,  enc_layers=2  batch_size=256    'lr': 0.0006, prediction':'simple  fusion':simple   参数少了接近一半
#04/07/2020 11:34:00 trainable params: 582,005
#04/07/2020 11:34:00 trainable params (exclude embeddings): 582,005
#[train.py:26]  experiment_times=5/5, test_score_sum=0.806512   下降明显


#LCQMC SCORE
#dev score:  BERT 89.4  ALBERT：87
#test score:  BERT 86.9  ALBERT：86.3

# 闲聊的匹配模型
#echo "python train.py" ${masker_ranker} "GPU:" ${gpu} "chat_courpus_log.txt"
CUDA_VISIBLE_DEVICES=${gpu} nohup python -u train.py ${config_json} ${host_name} > chat_courpus_log.txt 2>&1 &
tail -f chat_courpus_log.txt
# python train.py ${config_json} ${host_name}

#blocks=1,  enc_layers=2  batch_size=128 lr=0.001, prediction':'simple'    'fusion': 'full',  maxlen=20
# experiment_times=5/5, ave_test_score=0.802228, max_test_score=0.808940
# 如果只用lcqmc  experiment_times=5/5, ave_test_score=0.834672, max_test_score=0.841840  说明BQcorpus难度大   inference_time: 0.009

#batch_size，lr增加50%
#blocks=1,  enc_layers=2  batch_size=384   lr=0.001, prediction':'simple'    'fusion': 'full',  maxlen=20
#experiment_times=5/5, ave_test_score=0.795793, max_test_score=0.804216

# 降低学习率  准确率降低到75%-76%
#blocks=1,  enc_layers=2  batch_size=384   lr=0.0001, prediction':'simple'    'fusion': 'full',  maxlen=20

# 提高学习率 lr: 0.005 较好
#blocks=1,  enc_layers=2  batch_size=384   lr=0.005, prediction':'simple'    'fusion': 'full',  maxlen=20
#experiment_times=5/5, ave_test_score=0.803851, max_test_score=0.815180

#block=2 enc_layers=2 inference_time=19.337797 ms （慢了一倍） 81.7%  acc提高1%
#04/08/2020 10:12:37 Training time: 0:15:57.
#[train.py:29]  experiment_times=5/5, ave_test_score=0.817319, max_test_score=0.825341
#stats: {'inference_time': 1.1077707639851133, 'auc': 0.8672013396287949, 'f1': 0.786558313200088, 'acc': 0.7834477226134237}  微调后cpu测试

#block=2 enc_layers=1 inference_time=17.902684 ms （慢了接近一倍） 和enc_layers=2差不多
# experiment_times=5/5, ave_test_score=0.817765, max_test_score=0.828193
#experiment_times=5/5, ave_test_score=0.815269, max_test_score=0.821419
#GPU上测试  [demo.py:81]  stats: {'inference_time': 0.13252100063439243, 'auc': 0.9065379393588313, 'f1': 0.8247079924756113, 'acc': 0.8214190213031465}

# 接下来想要改成计算第一层block时enc_layers=2，将text1和若干text2拼接起来处理,后面enc_layers=1按照正常逻辑处理
# GPU上训练一个epoch的时间从1.97变成2.00  CPU上推理延迟降低从28.250353ms到22.8ms
#block=2 enc_layers=1 较好
#[train.py:30]  experiment_times=5/5, ave_test_score=0.820127, max_test_score=0.824048

#block=2 enc_layers=2  变差了   Training time: 0:09:01.
#[train.py:29]  experiment_times=5/5, ave_test_score=0.805981, max_test_score=0.810901


########   python demo.py configs/main.json5  cloudminds  benchmark-0
########   python demo.py configs/main.json5  wzk  benchmark-0

##   python evaluate.py $model_path $data_file



#04/08/2020 02:24:09 train loss: 0.4535 lr: 0.0050 gnorm: 0.5409 clip: 0
#04/08/2020 02:24:09 valid loss: 0.5331 inference_time: 471.0568 auc: 0.8262 f1: 0.7573 acc: 0.7329 cost_time: 5.2296 [NEW BEST]
#04/08/2020 02:26:46 > epoch 1 updates 300 loss: 0.3810 lr: 0.0050 gnorm: 0.1493
#04/08/2020 02:26:46 train loss: 0.3905 lr: 0.0050 gnorm: 0.2620 clip: 0
#04/08/2020 02:26:46 valid loss: 0.5112 inference_time: 469.1583 auc: 0.8408 f1: 0.7617 acc: 0.7501 cost_time: 7.8512 [NEW BEST]
#04/08/2020 02:29:26 > epoch 1 updates 400 loss: 0.3619 lr: 0.0050 gnorm: 0.2790
#04/08/2020 02:29:26 train loss: 0.3901 lr: 0.0050 gnorm: 0.3245 clip: 0
#04/08/2020 02:29:26 valid loss: 0.4931 inference_time: 480.2539 auc: 0.8442 f1: 0.7581 acc: 0.7555 cost_time: 10.5080 [NEW BEST]
#04/08/2020 02:32:07 > epoch 1 updates 500 loss: 0.4362 lr: 0.0050 gnorm: 0.2289
#04/08/2020 02:32:07 train loss: 0.4072 lr: 0.0050 gnorm: 0.2145 clip: 0
#04/08/2020 02:32:07 valid loss: 0.4736 inference_time: 477.7828 auc: 0.8575 f1: 0.7746 acc: 0.7679 cost_time: 13.2010 [NEW BEST]
#04/08/2020 02:34:46 > epoch 1 updates 600 loss: 0.3635 lr: 0.0050 gnorm: 0.1544
#04/08/2020 02:34:46 train loss: 0.3936 lr: 0.0050 gnorm: 0.1810 clip: 0
#04/08/2020 02:34:46 valid loss: 0.4765 inference_time: 479.9553 auc: 0.8611 f1: 0.7808 acc: 0.7721 cost_time: 15.8459 [NEW BEST]
#04/08/2020 02:37:29 > epoch 1 updates 700 loss: 0.3537 lr: 0.0047 gnorm: 0.1494
#04/08/2020 02:37:29 train loss: 0.3426 lr: 0.0049 gnorm: 0.1570 clip: 0
#04/08/2020 02:37:29 valid loss: 0.4909 inference_time: 483.9887 auc: 0.8640 f1: 0.7811 acc: 0.7728 cost_time: 18.5735 [NEW BEST]
#04/08/2020 02:37:49 > epoch 1 updates 750 loss: 0.3448 lr: 0.0047 gnorm: 0.2319


#04/08/2020 02:51:27 valid loss: 0.5831 inference_time: 472.1677 auc: 0.8149 f1: 0.7417 acc: 0.7283 cost_time: 5.4799 [NEW BEST]
#04/08/2020 02:54:14 > epoch 1 updates 300 loss: 0.3616 lr: 0.0050 gnorm: 0.1110
#04/08/2020 02:54:14 train loss: 0.3585 lr: 0.0050 gnorm: 0.1188 clip: 0
#04/08/2020 02:54:14 valid loss: 0.5229 inference_time: 515.2642 auc: 0.8297 f1: 0.7479 acc: 0.7445 cost_time: 8.2597 [NEW BEST]
#04/08/2020 02:56:57 > epoch 1 updates 400 loss: 0.3844 lr: 0.0050 gnorm: 0.0967
#04/08/2020 02:56:57 train loss: 0.3979 lr: 0.0050 gnorm: 0.1573 clip: 0
#04/08/2020 02:56:57 valid loss: 0.5151 inference_time: 472.3220 auc: 0.8452 f1: 0.7626 acc: 0.7564 cost_time: 10.9893 [NEW BEST]
#04/08/2020 02:59:48 > epoch 1 updates 500 loss: 0.3738 lr: 0.0050 gnorm: 0.0893
#04/08/2020 02:59:48 train loss: 0.3645 lr: 0.0050 gnorm: 0.1034 clip: 0
#04/08/2020 02:59:48 valid loss: 0.4932 inference_time: 508.3649 auc: 0.8555 f1: 0.7756 acc: 0.7612 cost_time: 13.8365 [NEW BEST]
#04/08/2020 03:02:37 > epoch 1 updates 600 loss: 0.3621 lr: 0.0050 gnorm: 0.1114
#04/08/2020 03:02:37 train loss: 0.3540 lr: 0.0050 gnorm: 0.1171 clip: 0
#04/08/2020 03:02:37 valid loss: 0.4740 inference_time: 479.0968 auc: 0.8608 f1: 0.7757 acc: 0.7690 cost_time: 16.6454 [NEW BEST]
#04/08/2020 03:05:26 > epoch 1 updates 700 loss: 0.2937 lr: 0.0047 gnorm: 0.1188
#04/08/2020 03:05:26 train loss: 0.3258 lr: 0.0049 gnorm: 0.1185 clip: 0
#04/08/2020 03:05:26 valid loss: 0.4598 inference_time: 475.4500 auc: 0.8695 f1: 0.7840 acc: 0.7809 cost_time: 19.4641 [NEW BEST]
#04/08/2020 03:07:11 > epoch 1 updates 750 loss: 0.3455 lr: 0.0047 gnorm: 0.0845
#04/08/2020 03:07:11 train loss: 0.3455 lr: 0.0047 gnorm: 0.0845 clip: 0
#04/08/2020 03:07:11 valid loss: 0.4817 inference_time: 507.1521 auc: 0.8635 f1: 0.7819 acc: 0.7684 cost_time: 21.2226 [BEST: 0.7809]
