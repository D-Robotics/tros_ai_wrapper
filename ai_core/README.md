# Build & Deploy

```bash
cd code
bash build.sh
cd -
```

# Run

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:s100/qat/script/aarch64/lib
cd s100/qat/script/aarch64/bin

# code/src/method/qat_bev_post_process_method.cc
./example --config_file=../../bev/bev_gkt_mixvargenet_multitask_nuscenes/workflow_latency.json
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
