# WMP → Isaac Lab 本机迁移

项目位于 `/home/qyw/WMP`；独立环境为 `wmp_lab`。运行时复用 `/home/qyw/MyHIMLoco/IsaacGymEnvs/IsaacLab` 的可编辑安装，不修改 `himloco` 或 MyHIMLoco 源码。原版 WMP 的 `legged_gym` 训练入口仍依赖 Isaac Gym；请使用本目录的入口。

## 启动

```bash
conda activate wmp_lab
cd /home/qyw/WMP

# 可视化训练：128 并行，完整混合地形
./isaaclab_migration/run.sh --num_envs 128 --iterations 1000 \
  --log_dir /home/qyw/WMP/isaaclab_migration/logs/my_training

# 无窗口训练
./isaaclab_migration/run.sh --num_envs 128 --iterations 1000 --headless \
  --log_dir /home/qyw/WMP/isaaclab_migration/logs/my_headless_training

# 从本次实测检查点继续训练（新增迭代数）
./isaaclab_migration/run.sh --num_envs 128 --iterations 1000 \
  --checkpoint /home/qyw/WMP/isaaclab_migration/logs/acceptance/model_144.pt \
  --log_dir /home/qyw/WMP/isaaclab_migration/logs/continued_training

# 可视化测试；--seconds 0 表示直到手动关闭窗口
./isaaclab_migration/run.sh --play --num_envs 4 --seconds 0 --follow_camera \
  --checkpoint /home/qyw/WMP/isaaclab_migration/logs/acceptance/model_144.pt \
  --log_dir /home/qyw/WMP/isaaclab_migration/logs/play_manual

# 查看训练曲线
/home/qyw/miniconda3/envs/wmp_lab/bin/tensorboard \
  --logdir /home/qyw/WMP/isaaclab_migration/logs --host 127.0.0.1 --port 6006
```

默认自由视角，鼠标可操作；`--follow_camera` 跟随首个机器人。GUI 训练按物理 200Hz、控制 50Hz、显示 10Hz 运行，实际墙钟速度取决于电脑。回放会按控制周期限速，渲染较慢时不会补跳物理步。4 环境满足算法初始化需求，分别放在不同地形列，减少画面中机器人重叠。

## 实际迁移范围

- 使用官方 WMP 仓库附带的 A1 URDF 和网格。显式维护 FR、FL、RR、RL 的策略关节顺序，映射至 Lab 的关节索引。
- `Core` 保留原版纯张量观测、奖励、PD 控制、地形课程、指令采样与 AMP 状态构造。CPU 测试逐方法比较 AST，避免意外改写原版计算。
- `WMPEnv` 重写场景建立、资产加载、状态同步、关节力矩写入、接触、重置和相机采集，继承 Isaac Lab `DirectRLEnv`。
- 保留原版 `ActorCriticWMP`、AMPPPO、AMP 专家动作、Dreamer RSSM 世界模型及深度预测器。没有把训练替换成普通盲走 PPO。
- 默认深度后端是 Isaac Lab `RayCasterCamera`：对完整三角网格投射 64×64 针孔射线，覆盖沟壑、台阶、倾斜通道和头顶横梁，限定只看地形，避免同地形上并行机器人污染观测。不是从二维高度扫描点插值出的图片。
- `--camera_backend rtx` 可使用真实 RTX `TiledCamera` 深度渲染。RTX 能看到其他环境机器人，且开销更大；共享地形上并行训练优先使用默认射线后端。RTX 平地相机遮挡探针已验证，完整混合地形训练验收使用射线后端。
- 相机位置 (0.27,0,0.03)m、水平视场58°、图像64×64、范围0–2m、每5个控制步更新、两帧FIFO延迟、归一化 `clip(depth_m,0,2)/2-0.5` 对齐 WMP。每个相机保留原版±5°固定安装俯仰扰动。
- 隐藏 URDF 中位于相机中心的视觉标记盒；RTX 从盒内采集会得到约1cm的无效图像。仅处理该视觉标记，原机器人碰撞、质量与关节仍来自同一资产。

## 16GB 显存配置

| 参数 | 原版 | 本机默认 |
| --- | --- | --- |
| 并行环境 | 4096 | 128 |
| 深度相机 | 1024 | 128（每个环境都有实际几何深度） |
| 图像 | 64×64 | 64×64 |
| 控制周期 | 0.02s | 0.02s |
| 世界模型批量×序列 | 16×64 | 2×16 |
| 每迭代世界模型更新 | 10 | 1 |
| 世界模型开始更新阈值 | 10000 | 256 |
| AMP 预加载 | 2000000 | 100000 |
| AMP replay | 1000000 | 100000 |
| 深度预测器批量/更新次数 | 1024/1000 | 32/1 |
| 地形难度行×类型列 | 10×20 | 10×20 |
| 地形边界 | 25m | 5m |
| 轨迹数据存储 | GPU+CPU | CPU，采样后传GPU |

保持原版世界模型网络宽度、PPO网络、AMP判别器和PPO更新轮数。小批量降低显存及单次计算，不保证相同样本效率。所有128个环境均有实际几何深度，因此保留深度预测器训练但无需为无相机环境合成图像。

## 验证与输出

```bash
python isaaclab_migration/tests/test_contract.py
./isaaclab_migration/run.sh --num_envs 16 --iterations 2 --terrain plane \
  --probe --headless --log_dir /home/qyw/WMP/isaaclab_migration/logs/probe_new

# 运行足够迭代采集完整轨迹，并要求世界模型发生更新
./isaaclab_migration/run.sh --num_envs 128 --iterations 100 --probe --require_wm_updates \
  --capture_dir /home/qyw/WMP/isaaclab_migration/outputs/new_capture \
  --log_dir /home/qyw/WMP/isaaclab_migration/logs/new_validation
```

`--probe` 检查关节映射、单环境重置不影响其他环境、超时、285维观测/30维AMP、接触数量及非遮挡深度。`--require_wm_updates` 检查世界模型更新计数及编码器权重变化。每轮报告记录整卡显存峰值，不仅统计 PyTorch。

检查点保存策略、PPO/AMP共同优化器、世界模型及优化器、深度预测器及优化器、判别器和AMP归一化器。恢复会严格加载模型并继续迭代计数；仿真状态、CPU轨迹缓存、随机数和环境课程状态会重新初始化，因此不是逐步完全相同的恢复轨迹。

每个日志目录含 TensorBoard、配置、检查点、`validation.json`、实际深度预览 `depth.png` 和米制深度 `depth_meters.npy`。可视化目录保存实际视口PNG；回放还可保存同一时刻深度小图，用 `python isaaclab_migration/scripts/make_preview.py /absolute/capture_dir` 生成动态图。

### 按检查点回看训练阶段

无窗口的大规模训练不会在保存权重时录视频。训练结束后，可以逐个载入保存的 `model_*.pt`，在固定的障碍和速度指令下重新回放，比较策略随训练轮次的变化。批量回放所有检查点的命令如下：

```bash
conda activate wmp_lab
cd /home/qyw/WMP
python isaaclab_migration/scripts/record_checkpoints.py \
  isaaclab_migration/logs/paper_2048 \
  --output_dir isaaclab_migration/outputs/paper_2048_checkpoints \
  --play_steps 1000 --terrain_col 10 --terrain_level 6 --forward_speed 0.5
```

查看某个阶段的视频：

```bash
xdg-open /home/qyw/WMP/isaaclab_migration/outputs/paper_2048_checkpoints/model_10000/preview.mp4
```

想现场打开仿真窗口，只回放一个检查点并录下画面，可以直接运行：

```bash
./isaaclab_migration/run.sh --play --num_envs 4 --camera_envs 4 \
  --checkpoint /home/qyw/WMP/isaaclab_migration/logs/paper_2048/model_10000.pt \
  --play_steps 1000 --follow_camera \
  --eval_terrain_col 10 --eval_terrain_level 6 --fixed_forward_speed 0.5 \
  --capture_every_steps 10 \
  --capture_dir /home/qyw/WMP/isaaclab_migration/outputs/my_gap_replay \
  --log_dir /home/qyw/WMP/isaaclab_migration/logs/my_gap_replay
```

结束后，将采样的 PNG 合成为视频：

```bash
python isaaclab_migration/scripts/make_preview.py \
  /home/qyw/WMP/isaaclab_migration/outputs/my_gap_replay \
  --frame_steps 10 --mp4_only
xdg-open /home/qyw/WMP/isaaclab_migration/outputs/my_gap_replay/preview.mp4
```

每个检查点的输出目录包含 `preview.mp4`、采样画面 PNG、同步深度图、`log/state.csv` 和 `log/validation.json`。视频录制为 1000 个控制步（20 秒仿真）、5 帧/秒，镜头跟随首个机器人。CSV 记录速度、前进距离、地形级别及重置标记；验证 JSON 记录检查点加载核对、运行步数和显存峰值。已录制的视频默认跳过；指定 `--overwrite` 会重新录制。

回放参数可按下面调整：

- `--only 10000` 只回放 `model_10000.pt`；不指定时处理目录中的全部 `model_*.pt`。
- `--play_steps 1000` 控制回放长度。A1 控制频率为 50 Hz，因此 1000 步等于 20 秒仿真；增加步数可以观察更长的地形通过过程。
- `--terrain_col 10` 选择障碍类型，`--terrain_level 6` 选择难度（0–9）。列 10 是沟壑，15 是攀爬，17 是倾斜通道，18 是低矮通道。
- `--forward_speed 0.5` 固定前进速度指令，单位为 m/s。比较不同检查点时应使用相同地形、难度、时长和速度。
- `--num_envs 4` 控制同时显示的机器人数量；只有首个机器人用于 CSV 指标统计。
- `--output_dir` 指定录像保存位置。换地形重新录制时使用不同目录，防止混淆。

例如，只在低矮通道回放最终策略：

```bash
python isaaclab_migration/scripts/record_checkpoints.py \
  isaaclab_migration/logs/paper_2048 --only 10000 \
  --output_dir isaaclab_migration/outputs/paper_2048_crawl \
  --terrain_col 18 --terrain_level 6 --forward_speed 0.5 --play_steps 1000
```

这些视频是从已保存检查点重新运行得到的评估回放，并非训练当时的原始画面。初始化地形、命令及录像参数保持一致，物理仿真仍有随机性，不能保证逐帧相同。训练与 GUI 回放同时使用显卡可能挤占 16 GiB 显存，因此建议训练结束后批量生成。

### 本次 10000 轮训练的阶段回放

2026-10-18 使用第 6 级间隙地形、0.5 m/s 前进指令、1000 步（20 秒仿真）回放了全部保存检查点。视频在 5 或 10 帧/秒下均为 20 秒，镜头持续跟随首个机器人。下表的“最远前进”和“重置”来自对应 `log/state.csv`，表示一次仿真结果，适合查看训练阶段趋势，不是统计成功率。

| 迭代检查点 | 视频 | 最远前进 | 首个机器人重置 |
| ---: | --- | ---: | ---: |
| 1 | [preview.mp4](outputs/paper_2048_checkpoints/model_1/preview.mp4) | -0.41 m | 0 |
| 1001 | [preview.mp4](outputs/paper_2048_checkpoints/model_1001/preview.mp4) | 2.06 m | 2 |
| 2001 | [preview.mp4](outputs/paper_2048_checkpoints/model_2001/preview.mp4) | 2.33 m | 4 |
| 3001 | [preview.mp4](outputs/paper_2048_checkpoints/model_3001/preview.mp4) | 4.44 m | 2 |
| 4001 | [preview.mp4](outputs/paper_2048_checkpoints/model_4001/preview.mp4) | 11.65 m | 0 |
| 5001 | [preview.mp4](outputs/paper_2048_checkpoints/model_5001/preview.mp4) | 10.67 m | 0 |
| 6001 | [preview.mp4](outputs/paper_2048_checkpoints/model_6001/preview.mp4) | 10.37 m | 1 |
| 7001 | [preview.mp4](outputs/paper_2048_checkpoints/model_7001/preview.mp4) | 10.80 m | 0 |
| 8001 | [preview.mp4](outputs/paper_2048_checkpoints/model_8001/preview.mp4) | 10.07 m | 0 |
| 9001 | [preview.mp4](outputs/paper_2048_checkpoints/model_9001/preview.mp4) | 11.65 m | 0 |
| 10000 | [preview.mp4](outputs/paper_2048_checkpoints/model_10000/preview.mp4) | 11.08 m | 0 |

最终策略在其他地形的录像也保存在 `outputs/paper_2048_{climb,tilt,crawl}/model_10000/preview.mp4`。这次单次回放结果：攀爬地形最远 4.68 m、重置 1 次；倾斜通道最远 5.15 m、重置 1 次；低矮通道最远 10.25 m、没有重置。每个目录都有对应 `log/state.csv` 和 `log/validation.json`。

## 物理差异与 D435i 后续

Isaac Gym 与 Isaac Lab/Sim 的 PhysX、URDF导入、接触求解不同，无法保证相同动力学轨迹。本迁移用 Lab 足部净接触力替代 Gym 足部力传感器读数；采用相同奖励公式但底层测量实现不同。现有短训练只验收数值稳定、梯度更新、深度链路、保存恢复及可视化，不能证明论文级越障性能已复现。

当前相机是理想几何传感器。D435i 接入前需用真实标定内参/安装外参替换针孔参数，处理米制单位、无效深度、畸变/裁剪/缩放与延迟，并加入符合实测的噪声/丢帧随机化。这里没有连接实物 D435i，也没有进行实机部署验证；D435i 的 IMU 不等于已标定的机器人本体状态。

环境是 `himloco` 的独立克隆，新增 `ruamel.yaml==0.17.40` 和 `pybullet==3.2.7`。不要在该环境执行根目录旧版 `requirements.txt`。

## 本次实测记录（2026-10-16）

128 并行完整地形：初始100次迭代，随后严格恢复20+20+4次，累计144次PPO/AMP迭代，442,368个环境控制步；世界模型累计130次更新、深度预测器11次。单轮新训练测得世界模型编码器最大权重变化均大于零。

有效可视化训练运行的整卡显存峰值最高7001 MiB（约6.84 GiB）；最终验收为6974 MiB。早期失败的RTX调试运行有多个进程重叠，其峰值没有用作默认参数的依据。

最终检查点：`/home/qyw/WMP/isaaclab_migration/logs/acceptance/model_144.pt`。4项CPU测试及GPU部分重置、超时、关节映射、接触数量、深度、Runner缓冲区写入探针通过。最终恢复前后参数张量精确相等；输出按FP32的rtol=1e-5、atol=1e-5核对，具体误差写入报告。

查看 `logs/acceptance/validation.json` 与 `logs/playback_verified/validation.json` 获取训练和回放的实际证据。`outputs/playback_verified/preview.gif` 是真实Lab视口和同期深度图合成的回放，`state.csv`保存在回放日志目录。

最终回放实测：4环境、491步、9.82秒仿真时间，0重置；输出核对误差为0。首个机器人指令约0.274m/s，实际平均前向速度约0.0049m/s，尚未形成有效速度跟踪，不应当作为收敛策略使用。最终RTX后端16环境平地回归也通过，整卡峰值4745MiB。

`environment.yml`是本机环境清单，包含本地可编辑Isaac Lab路径及pip依赖；不应视为跨机器完整锁文件。在本机新建同类环境优先克隆已验证的`himloco`，然后补充上述两个依赖。

## 1024 并行论文参数配置

`./isaaclab_migration/paper_1024.sh` 固定 1024 个 A1，并使用 256 路深度图（论文 4096 个 A1 中有 1024 路，保持 25% 覆盖比例）。其中倾斜和低矮通道环境全部分配相机，其余相机从其他地形随机选取；无相机环境在训练世界模型时使用深度预测器。脚本恢复原版世界模型 `16×64`、每轮 10 次更新、10000 步预热，AMP 200 万预加载与 100 万 replay，以及深度预测器每 10 轮执行 `1024×1000` 次训练。网络、FP32、PPO 每环境 24 步、20 秒回合均沿用原版。奖励课程的迭代阈值乘 4，使经验步数与 4096 并行原版对齐。

```bash
cd /home/qyw/WMP
./isaaclab_migration/paper_1024.sh --iterations 40000
# 相当于原版 4096 并行训练 10000 轮的环境步数；继续到 80000 轮对应原版 20000 轮。
```

该脚本只提供训练配置，不会自动开始长训练。`--iterations` 是本次调用额外执行的轮数；带 `--checkpoint /absolute/model_*.pt` 继续训练时，检查点内的世界模型与训练配置会覆盖命令行对应项。续训应保留 `--reward_schedule_scale 4`，并用新日志目录保存。检查点不会恢复仿真、随机数和 CPU 轨迹缓存，因此严格数值连续性不成立。

实测：1024 并行、256 路几何深度、上述完整超参数训练 21 轮通过；世界模型 130 次更新、深度预测器 2000 次更新，世界模型从无相机环境提取了 1605 个预测深度样本。整卡显存峰值 7730 MiB。早期策略频繁重置，前 21 轮实际采样序列平均 12.6 帧、最长 29 帧，尚未达到设置的 64 帧；需更长训练评估这一点。

1024 路 RTX 渲染深度在本机 31.5 GB 内存下被 Linux OOM 杀掉（进程匿名内存约 22 GB），因此此配置使用 Isaac Lab 的几何光线投射深度。此后端保留 64×64 分辨率、58° 视场、安装位姿、10 Hz 和 100 ms 延迟，但只与地形网格相交，没有 RTX 自遮挡、材质与 D435i 噪声。Isaac Gym→Isaac Lab 的动力学差异也仍需通过长训练和实机评估。1024 并行的每轮 PPO 经验和每小批量样本是论文 4096 并行的四分之一，40k/80k 轮只是环境步数等量，不能保证论文性能等量。

### 2048 环境、512 路深度

`./isaaclab_migration/paper_2048.sh` 将相机覆盖比例保持为 25%，每环境收集 48 步，使每轮 PPO 总样本数 `2048×48=98304` 与论文 `4096×24=98304` 相同。默认 10000/20000 轮分别与论文有相同总环境步数和 PPO 更新轮数；奖励课程阈值沿用原版，世界模型、AMP、深度预测器超参数也沿用原版。单个机器人的一次 rollout 从论文 24 步变成 48 步，时间相关性会不同。若继续训练，仍需显式保留 `--reward_schedule_scale 1`。

```bash
cd /home/qyw/WMP
./isaaclab_migration/paper_2048.sh --iterations 10000
```

2048/512、24 步的 21 轮完整配置实测通过：世界模型 150 次更新、深度预测器 2000 次更新、预测深度采样 1807 次，`16×64` 满批量反向传播通过；整卡峰值 10210 MiB，进程峰值内存 12760 MiB。24 步的 100 轮资源诊断最高 11682 MiB 显存、12551 MiB 进程内存，实际采到了完整 64 帧。48 步完整配置另测 21 轮通过：世界模型 180 次、深度预测器 2000 次、预测深度采样 2166 次；整卡峰值 10724 MiB，进程峰值内存 12558 MiB。48 步在早期实际采样最长 57 帧。RTX 路径在 1024 环境、128 路相机时已经触发系统内存 OOM，因此 2048/512 正式脚本仍使用几何深度；RTX 依靠 GPU 渲染，但 Isaac Sim 同时消耗大量系统内存。

若更看重保持论文每环境 24 步，可运行 `./isaaclab_migration/paper_2048.sh --rollout_steps 24 --reward_schedule_scale 2 --iterations 20000`；这会使 PPO 每轮总样本和每轮环境步数变为论文的一半，而同等环境步数下的 PPO/世界模型更新次数变为两倍。两种方案均只做了短期稳定性验证，不能由此推断收敛质量或最终论文成绩。

## ROBOCON 梅花桩、轮胎阵、高墙课程

新增三类各10级的课程，9级为标准尺寸，支持单项、三类合训与原地形混合。课程成功率、标准强化采样、规则判定及使用命令见 [ROBOCON.md](ROBOCON.md)。

只检查地形、不创建机器人或训练器：

```bash
./isaaclab_migration/preview_terrains.sh --level 9
```

在预览面板选择类型、0–9级并点击聚焦，可对比课程变化。确认地形后再使用 `run.sh --terrain robocon --competition_stage learn` 开始训练。
### 手动录制每 300 轮的赛道回放

训练仍按每 300 轮保存检查点。需要回放时，在系统终端执行：

```bash
./isaaclab_migration/record_course_checkpoints.sh /训练日志目录 1500
```

省略轮次会处理该目录下所有尚未完成的 300 轮检查点；先加 `--list` 可列出可用轮次。每个检查点单独录制梅花桩、轮胎阵、高墙的 L0 与 L9，共六段。视频为 20 fps、20 秒，与 50 Hz 仿真时间一致；相机位于机器人正后方朝向赛道。完成的视频和索引位于训练目录的 `checkpoint_videos/`，中断后重启脚本会跳过已验证完成的片段。运行回放前留出 GPU/内存余量，避免与大型训练同时占满资源。
