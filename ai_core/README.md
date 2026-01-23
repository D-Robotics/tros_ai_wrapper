# Prepare the Back-Injection Dataset

Run the following commands in the terminal of the RDK system to download and unzip the dataset:

```shell

# Board-side dataset download, hobot_bev 
cd ~
wget http://archive.d-robotics.cc/TogetheROS/data/nuscenes_bev_val/nuscenes_bev_val.tar.gz

# Unzip
mkdir -p ~/hobot_bev_data
tar -zxvf ~/nuscenes_bev_val.tar.gz -C ~/hobot_bev_data


# Download the point cloud file for back-injection on the board side, hobot_centerpoint
cd ~
wget http://archive.d-robotics.cc/TogetheROS/data//hobot_centerpoint_data.tar.gz

# Unzip
mkdir -p ~/centerpoint_data
tar -zxvf ~/hobot_centerpoint_data.tar.gz -C ~/centerpoint_data
 ```

# Build & Deploy

```bash
# s100
cd code
bash build_all.sh -p s100
cd -

# s600
cd code
bash build_all.sh -p s600
cd -
```

# Run s100

```bash
# s100 
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:s100/qat/script/aarch64/lib
cd s100/qat/script/aarch64/bin

# hobot_bev
ln -s ~/hobot_bev_data/nuscenes_bev_val nuscenes_bev_val

# code/src/method/qat_bev_post_process_method.cc
./example --config_file=../../bev/bev_gkt_mixvargenet_multitask_nuscenes/workflow_latency.json

# hobot_centerpoint
ln -s ~/centerpoint_data centerpoint_data
# 替换centerpoint_data文件夹中的nuscenes_lidar_val.lst文件，用script目录下的nuscenes_lidar_val.lst文件替换
./example --config_file=../../detection/centerpoint_pointpillar_nuscenes/workflow_latency.json
```


# Run s600（model 来源 drobotics_s100_s600_open_explorer_v3.7.0 ）

```bash
# s600 
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:s600/qat/script/aarch64/lib
cd s600/qat/script/aarch64/bin

# hobot_bev
ln -s ~/hobot_bev_data/nuscenes_bev_val nuscenes_bev_val

# code/src/method/qat_bev_post_process_method.cc
./example --config_file=../../bev/bev_gkt_mixvargenet_multitask_nuscenes/workflow_latency.json

# hobot_centerpoint
ln -s ~/centerpoint_data centerpoint_data
# 替换centerpoint_data文件夹中的nuscenes_lidar_val.lst文件，用script目录下的nuscenes_lidar_val.lst文件替换
./example --config_file=../../detection/centerpoint_pointpillar_nuscenes/workflow_latency.json
```

# Data Preprocess

[自动驾驶nuScenes数据集-Mini版](https://aistudio.baidu.com/datasetdetail/157586)

[nuScenes v1.0-trainval01](https://aistudio.baidu.com/datasetdetail/182741)

```bash
# mini dataset
python3 bev_preprocess.py --model=bev_gkt_mixvargenet_multitask_nuscenes --data-path=data/nuScenes --meta-path=data/nuScenes --reference-path=../../script/config/reference_points --save-path=./nuscenes_bev_val

# train dataset
python3 bev_preprocess.py --model=bev_gkt_mixvargenet_multitask_nuscenes --data-path=data/nuScenes/ --meta-path=data/nuScenes/v1.0-trainval/v1.0-trainval_meta --reference-path=../../script/config/reference_points --save-path=./nuscenes_bev_val
```
