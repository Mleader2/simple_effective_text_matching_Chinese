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
#tail -f chat_courpus_log.txt
#python train.py ${config_json} ${host_name}

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


#block=2 enc_layers=1 inference_time=17.902684 ms （慢了接近一倍）

# 接下来想要改成计算第一层block时enc_layers=2，将text1和若干text2拼接起来处理,后面enc_layers=1按照正常逻辑处理
########   python demo.py configs/main.json5  cloudminds

##   python evaluate.py $model_path $data_file