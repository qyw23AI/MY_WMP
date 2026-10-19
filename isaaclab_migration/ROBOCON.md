# ROBOCON 三类课程地形（Isaac Lab / WMP / A1）

本次实现仅扩展当前实际运行的 `isaaclab_migration`；不更改旧 Isaac Gym 入口。原 `--terrain mixed` 默认行为保持。新增配置必须显式选择。策略、AMP、深度世界模型及285维观测接口保留。

## 方案与标准尺寸

三类地形均有10级（0–9），第9级明确使用标准尺寸，不使用旧课程的 `行号/总行数` 插值作为终点。几何函数由训练地形与纯地形预览共用。

| 地形 | 0→9级课程 | 第9级 | 严格成功 |
| --- | --- | --- | --- |
| 梅花桩 plum_stones | 桩面40→20cm，高5→20cm，纵向布局逐渐展开 | 六个20cm立方体，两端80×80×20cm平台，总长3.5m，同排净距40cm，排间净距20cm | 从平台前地面出发，两端平台及六桩均有足底支撑接触，完整离开末平台并稳定落地；两足落地或侧面掉落失败 |
| 轮胎阵 tire_array | 高3→16.5cm，逐渐撤去孔内教学填充，6–9级无填充 | 外径55cm/内径33cm，轴向厚16.5cm，2×4，总长2.2m/宽1.1m | 从短边进入、对侧短边离开并稳定落地；两足接触阵外地面失败；孔内地面接触合法 |
| 高墙 high_wall | 高3/5/8/11/14/17/20/24/27/30cm，宽厚固定 | 高30cm、宽80cm、厚8cm | 机身与四足完整越过，无侧绕，落地稳定 |

轮胎是理想化固定刚体，不包含橡胶形变。16.5cm来自规则胎宽，真实平放高度仍需实测。标准几何对齐本次修正的 `output/scene_audit/robocon2027.v2.xml`。训练使用静态三角网格，环形孔洞不做凸包填充。

## 课程、奖励和出生

- `learn`：每类地形有独立难度前沿，初始0级。每级收集至少50次尝试；窗口成功率≥80%升一级，<40%降一级，最高级保持9级。梅花桩0–2级可先学习通过，3–9级必须六桩全覆盖；严格成功率始终单独记录全覆盖结果。
- `standard`：每次新回合按60%第9级、25%第7–8级、15%第0–6级采样。这是显式强化模式，不会在未经确认时自行启动。
- `eval`：固定最高级；`--play --eval_terrain_level ... --eval_terrain_col ...`可以覆盖到指定等级和列，重置不升降级。比赛验收必须用9级，分别统计成功率，不用平均地形等级代替。
- 新增最大前进距离增量奖励、每桩一次接触奖励和完成/违规事件奖励，保留原WMP速度跟踪与AMP奖励。不设置实体侧护栏；通过足部/机身相对赛道的位置判定绕行并施加惩罚。不向actor输入下一落脚点或桩覆盖记录。
- 接触判定使用A1足球半径2cm估计足底位置，支撑力>1N；桩顶接触需要位置和高度同时匹配。完成需机身与四足越过终点，并满足至少两足支撑、姿态稳定和竖直速度约束，持续0.3秒。
- 梅花桩从入口平台前0.7m的地面出生，并记录入口/出口平台顶面支撑接触；前腿上平台而后腿仍在入口前地面属于正常接近过程。跳过入口或出口平台不能算成功。轮胎、高墙从前方安全地面出生。出生前后扰动±2.5cm、近默认关节姿态，初速度为0。三类地形先不施加原版随机推力。梅花桩完整任务包括接近、上平台、过桩、经过末平台及下平台。
- 三类地形前进指令0.15–0.30m/s，侧向指令为0、朝向沿+X。高墙允许墙面接触，保留倒地/机身过低/超时判定，不把每次机身碰墙立即判失败。
- 精确几何查询代替新增地形的10cm粗高度网格；查询与分片三角曲面一致，保持187/525点接口及原网络维度。
- TensorBoard按类型记录 `success`、`strict_success`、`coverage`、`failure`、`timeout`、`frontier`、`standard_attempts`、`standard_success`。每种/每级累计尝试和成功次数存储在课程状态中。
- 新检查点附带课程前沿与计数，续训恢复；旧检查点仍可加载并从初始课程开始。仿真状态、轨迹缓存和随机数状态依旧不会完整恢复。

## 纯地形检查：本次停在这里

```bash
cd /home/qyw/WMP
./isaaclab_migration/preview_terrains.sh --level 9
```

此入口不创建机器人、策略、优化器或训练器。展示三种类型×10级共30个地形。每一行固定一种类型，同一行从左向右为0–9级；相邻单元中心间距8m，与训练地形的X难度轴、Y类型轴一致。进入窗口默认显示完整三行课程。蓝色地面圆点标记机器人机身出生位置的XY投影（标记仅用于预览，不属于训练碰撞地形）。面板可选类型和等级，点击：

- `Focus selected course`：聚焦选中的类型/等级。
- `Show three courses at selected level`：对比同一等级三种地形。
- `Overview: all 10 levels`：查看三行完整课程排列。
- `Show selected type: level 0 to 9`：只显示所选类型的一行课程。

场景节点路径 `/World/courses/<type>/level_<0-9>`。鼠标操作沿用Isaac Sim视口。关闭窗口即可退出。

输出：`outputs/terrain_inspection/courses.json`包含完整参数；GUI启动成功后保存 `curriculum_layout.png`（完整排列）和 `standard_courses.png`（最高级三种地形）。

## 检查命令（无训练）

```bash
/home/qyw/miniconda3/envs/wmp_lab/bin/python isaaclab_migration/tests/test_contract.py
/home/qyw/miniconda3/envs/wmp_lab/bin/python isaaclab_migration/tests/test_competition.py
/home/qyw/miniconda3/envs/wmp_lab/bin/python isaaclab_migration/scripts/check_competition_env.py --headless
```

CPU检查包括原WMP张量方法一致性、标准尺寸、修正XML桩位、所有等级网格、独立垂直射线对比高度查询、地面出生、上平台过程、禁止跳过平台、同类型难度行、落脚规则、禁止侧绕、严格完成、课程升降级/采样和课程状态序列化。环境检查只运行零动作仿真，不创建策略或优化器；覆盖A1观测/深度/高度查询、等级冻结与局部重置。

## 用户确认地形后再执行的训练入口

下面只提供命令，实施本次请求时不执行训练。

```bash
# 单独验证三类地形；也可将--terrain改为plum_stones/tire_array/high_wall
./isaaclab_migration/run.sh --terrain robocon --competition_stage learn \
  --num_envs 128 --camera_envs 128 --headless --iterations 1000 \
  --log_dir /home/qyw/WMP/isaaclab_migration/logs/robocon_learn

# 单项形成能力后，显式启动标准强化；检查点路径换为实际生成的文件
./isaaclab_migration/run.sh --terrain robocon --competition_stage standard \
  --num_envs 128 --camera_envs 128 --headless --iterations 1000 \
  --checkpoint /absolute/path/to/model.pt \
  --log_dir /home/qyw/WMP/isaaclab_migration/logs/robocon_standard
```

`robocon`类型列为：0–8梅花桩、9–15轮胎、16–19高墙。

`robocon_mixed`有40列：旧地形22列（55%）、梅花桩8列（20%）、轮胎6列（15%）、高墙4列（10%）。`robocon`训练从1024环境中抽取256个使用真实深度，其余环境使用深度预测器；`robocon_mixed`还需给全部倾斜/低矮通道环境配置真实深度，相机数量须满足这个下限。单项配置各4列。

从原WMP检查点微调时继续使用 `--checkpoint`，网络形状保持兼容。首轮训练后才能判断超参数、奖励比例和A1标准高墙通过能力；本次检查不证明已经学会越障。

## 2026-10-19：早期1024环境训练记录（历史配置）

`train_robocon_1024.sh` 是早期训练入口，配置不代表后续2048环境无侧护栏训练。默认首阶段新增10000轮，可用 `ITERATIONS` 配置，使用全新日志目录并拒绝覆盖既有配置。从 `paper_2048/model_10000.pt` 微调；旧编号10000对应本次0轮，本次300轮保存 `model_10300.pt` 并立刻暂停优化录制，而非训练结束再录像。

资源配置保持原网络、AMP、深度预测器和世界模型/优化器权重。1024环境中的256个使用64×64真实几何raycast深度，其余使用深度预测器；rollout48，世界模型批次4×64、每轮10次更新、启动阈值256（适配初期较短的失败回合），AMP预载/回放各100000；深度预测器每10轮更新1次，批次32。相对于旧检查点，显式缩小世界模型批次和AMP缓存，并降低深度预测器更新次数；未删除训练模块。训练回合上限45秒，保证0.15m/s慢速指令有时间从地面出发完成约5m的梅花桩任务；视频仍各20秒。CPU深度历史使用float32文件映射缓存 `replay/*.f32`，由OS回收页面，避免常驻占满32GB宿主内存。

首次GUI预检因1024机器人复制非实例网格而触发宿主OOM（内核记录约18GB进程内存）。改为共享机器人网格实例，保留动力学/碰撞配置；直接在共享视觉源隐藏原相机遮挡盒，base视觉/碰撞保持实例化；并回收初始化临时内存。纯平地改为精确的两三角形，课程障碍网格不变。本次 `--compact_courses` 使用3列（每类一列），各10级；1024环境约等分三类，不再复制20列相同障碍。只隐藏其余1023机器人的显示，物理与相机均运行，录像观察env0。

每300轮同进程生成6个20秒MP4：三类各含当时课程前沿、固定9级。固定种子2027、前进指令0.20m/s，字幕标记类型/等级/新增轮次/旧编号。`videos/index.md` 和 `videos/index.json` 即时更新。评估使用推理模式、无优化器调用、无训练经验写入；结束恢复课程累计状态和随机数、重新开始仿真回合及未完成世界模型轨迹。完成的WM数据集、AMP回放、网络和优化器一直留在原进程。**这不是完整仿真状态的连续恢复**，每300轮会丢弃正在进行的回合/未完成轨迹；进程异常恢复还会丢失RAM中的经验，仅能从checkpoint恢复模型/优化器/课程计数。

稳定入口：`console.log`、`pid`、`config.json`、`progress.json`、`events.out.tfevents.*`（TensorBoard）、`videos/index.md`。失败写入 `failure.json`，最终验证写入 `validation.json`。`progress.json` 包含本次轮次、绝对checkpoint迭代号、WM累计更新和每类每级计数。课程 `totals[type,level,:]` 的12项依次为：失败、违规、超时、入口平台、出口平台、六个独立桩接触、六桩全覆盖；`attempts`/`wins`/`strict_wins` 提供分母和成功次数，9级成功率使用 `strict_wins[:,9]/attempts[:,9]`，不能由前沿推断成功率。
