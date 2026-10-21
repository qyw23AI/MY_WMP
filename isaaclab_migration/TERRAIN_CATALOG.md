# 当前可训练地形目录

本文档按 `wmp_lab` 当前训练入口和地形生成代码整理，记录可以通过 `isaaclab_migration/run.sh --terrain ...` 选择的训练地形。它区分两套模式：`mixed` 是原有通用地形混合课程；其它障碍类型通过竞赛课程配置器单独训练或组合训练。可视化预览器支持的内容不一定都能直接作为 `--terrain` 参数。

## 1. 地形与训练入口总览

| 类别 | 训练参数 | 等级 | 说明 |
| --- | --- | ---: | --- |
| 通用混合地形 | `mixed` | 10行课程 | 默认地形课程，包含波浪、坡地、上/下楼梯、离散障碍、沟壑、坑、倾斜通道、低矮通道和随机粗糙平地。 |
| ROBOCON竞赛总混合 | `robocon` | 梅花桩15级；其余10级 | 9列组合梅花桩、7列轮胎阵，其余7种竞赛类型各4列。 |
| ROBOCON兼容混合 | `robocon_mixed` | 梅花桩15级；其余10级 | 保留旧课程中若干通用地形列，并加入竞赛地形；比例见本文件下文。 |
| 楼梯沟壑组合混合 | `gap_combos` | 10级 | 仅混合 `stair_gap` 与 `stair_gap_ramp`，每类4列。 |
| 单项竞赛地形 | 下表各参数 | 依地形而定 | 可用 `--terrain <类型>` 独立训练。 |
| 平地 | `plane` | 无课程等级 | 无地形障碍，用于平地训练/调试。 |

旧参数 `platform_plum_combo` 仍可用，映射到当前 `plum_stones`（带进出口高台的组合梅花桩）。

## 2. 单项竞赛地形

坐标描述中，赛道沿前方直行，长度单位为米，角度单位为度。凡注明“深沟”的赛道，训练碰撞地形的沟底为Z=−5m，机器人落入沟底或侧沟按任务规则重置。

| 训练参数 | 等级/标准级 | 地形结构与课程项 | 主要通过/失败规则 |
| --- | --- | --- | --- |
| `plum_stones` | L0–L14；标准L10 | 平地起步→80×80×20cm高台→6根交错梅花桩→同尺寸出口高台→平地。赛道宽80cm、两侧深沟；固定20×20×20cm桩，仅相邻桩纵向/横向净间隔和进出口净间隔学习。 | 六根桩均需取得顶面支撑；任何脚在桩间、进出口间隙或侧沟落低即失败。完成高台、桩区和离场并稳定支撑。L0相邻交错桩纵向投影重叠7.5cm、横向净间隔2.5cm；L10为比赛基准10/20cm；L14为15/25cm。 |
| `plum_stones_flat` | L0–L14；标准L10 | 平地道路顶面和桩顶同为Z=20cm；中间由6根固定20×20×20cm交错桩组成，桩间挖低20cm；道路宽80cm，两侧深沟。仅桩间距及进出口正间隔学习。 | 与组合梅花桩相同，需要覆盖全部六根桩；脚落入桩间槽、进出口缝或侧沟则重置。进出平地不要求上下台。 |
| `tire_array` | L0–L9；标准L6 | 2行×4列固定轮胎，外径55cm、内径33cm，孔洞始终开放；胎面有固定比例浅凹凸网格。仅轮胎高度变化，3–19.8cm。 | 连续向前通过轮胎阵；轮胎孔内仍是可接触的地面。轮胎为固定几何，不模拟橡胶形变。 |
| `high_wall` | L0–L9；标准L6 | 横跨80cm赛道的墙，厚8cm；墙高5–36cm，课程由低至高。赛道两侧为深沟。 | 需越过横墙继续前进；撞墙、跌落或进入侧沟触发重置。 |
| `platform` | L0–L9；标准L9 | 一个80×80cm高台，高度5–30cm；从地面上台再从另一侧下台。 | 一次完整上台、越台、下台任务；侧面深沟/越界落地失败。 |
| `ramp` | L0–L9；标准L9 | 直线上坡→80cm长平台→直线下坡；坡角5–30°，坡段水平投影各80cm，赛道宽80cm。 | 沿前进方向完整上坡、越过平台并下坡。无转弯。 |
| `platform_gap` | L0–L9；标准L9 | 6个同高80×80cm平台构成直线跳台路径，台间净间隙5–45cm；平台顶面Z=0，周围/下方为Z=−5m深沟。 | 逐台跳跃。允许四足先后落地；任何脚踩入间隙、卡在两台之间或落到赛道侧面即失败并重置。 |
| `stair_gap` | L0–L9；标准L6 | 平地→两个独立踏步→80×80cm起跳台→跨沟→80×80cm落地台；直线，赛道宽80cm，侧沟及缺口底Z=−5m。 | 依次获得踏步、起跳台支撑并跳到相邻落地台；空中脚不判失败，落在目标台外/间隙/侧沟失败。四足可先后接触，完成规定接触后在落地台稳定支撑0.3秒结束。 |
| `stair_gap_ramp` | L0–L9；标准L6 | 与 `stair_gap` 相同，落地后接一段直线下坡并进入终点平地。坡角5–30°，坡底接Z=0赛道。 | 完成楼梯、起跳、落地后继续通过下坡；必须实际接触坡面并在坡底平地稳定支撑0.3秒。全程不转弯。 |

### 竞赛课程关键等级

| 课程 | L0 | 标准/比赛参考 | 最高等级 |
| --- | --- | --- | --- |
| 两种梅花桩 | 纵向净距−7.5cm、横向净距2.5cm、进出口净距1.5cm | L10：10/20/10cm | L14：15/25/15cm |
| 轮胎阵 | 高3cm | L6：高16.5cm | L9：高19.8cm |
| 高墙 | 高5cm | L6：高30cm | L9：高36cm |
| 单高台 | 高5cm | L9：高30cm | L9：高30cm |
| 上下坡 | 坡角5° | L9：坡角30° | L9：坡角30° |
| 平台跳跃 | 间隙5cm | L9：间隙45cm | L9：间隙45cm |
| 楼梯沟壑组合 | 每级踏步4cm、总高12cm、沟宽5cm | L6：每级12cm、总高36cm、沟宽36cm；带坡版为20° | L9：每级18cm、总高54cm、沟宽54cm；带坡版为30° |

除梅花桩外，竞赛地形均为10级。`learn` 阶段按成功率升降级；`standard` 对新强化障碍主要采样标准参考、最高等级和二者间等级；`eval` 可指定类型和等级。课程升降阈值、统计状态和检查点兼容细节见 [OBSTACLE_CURRICULUM.md](OBSTACLE_CURRICULUM.md)。

## 3. 通用混合地形 `mixed`

`mixed` 使用10行难度课程。地形列按 `terrain_proportions` 分配，生成器中的类型与含义如下。随机高度与障碍尺寸随课程难度逐步增加，部分类别同时叠加±5cm随机粗糙度。

| 内部类型 | 地形内容 | 生成特点 |
| --- | --- | --- |
| `wave` | 起伏波浪地 | 5个波峰/波谷，并叠加随机粗糙高度。 |
| `slope` | 上坡或下坡 | 金字塔式斜坡；不同列选择上坡或下坡，叠加随机粗糙高度。 |
| `stairup` | 上楼梯 | 金字塔式台阶；台阶高度随等级增加。 |
| `stairdown` | 下楼梯 | 下行台阶，台阶高度随等级增加。 |
| `discrete` | 离散方块障碍 | 地面上随机布置矩形块，并叠加随机粗糙高度。 |
| `gap` | 沟壑跳跃 | 中间断开一段路面，沟宽随难度增长；沟底按原生成器高度为−5m，并叠加粗糙度。 |
| `pit` | 深坑/攀爬台 | 路面中设置两段高于坑底的横向区域，高度随难度增长，并叠加随机粗糙度。 |
| `tilt` | 倾斜通道 | 使用斜置的通道/支撑网格，横向通过区域随难度变化。 |
| `crawl` | 低矮通道 | 低矮横梁限制通过高度，结构在地形生成器中固定。 |
| `roughflat` | 随机粗糙平地 | 平坦路线叠加小尺度随机高度；保留现有粗糙地面课程。 |

默认 `LeggedRobotCfg` 的混合比例为 `wave/slope/stairup/stairdown/discrete/gap/pit/tilt/crawl/roughflat = 10/10/30/25/15/10/0/0/0/0%`。A1配置采用 `0/5/15/15/0/25/25/5/5/5%`。最终比例取所选机器人配置；可由命令行 profile 覆盖。

## 4. 混合竞赛配置的地形构成

`robocon` 总计44列：梅花桩9列、轮胎阵7列，以及高墙、高台、坡道、平台跳跃、楼梯沟壑组合A、带下坡组合B、平地梅花桩各4列。每列代表对应地形的一组课程等级。

`robocon_mixed` 同时保留部分旧通用类型：包含坡地、上/下楼梯、沟壑、坑、倾斜通道、低矮通道和粗糙地面，并额外加入两种梅花桩、轮胎阵和全部其它竞赛地形。具体列数直接定义在 `configure_competition()` 中。

`gap_combos` 只包含 `stair_gap` 和 `stair_gap_ramp` 两种直线组合地形，各4列、10级课程。

## 5. 使用方法

独立训练某种竞赛地形：

```bash
./isaaclab_migration/run.sh --terrain plum_stones --competition_stage learn
./isaaclab_migration/run.sh --terrain platform_gap --competition_stage learn
./isaaclab_migration/run.sh --terrain stair_gap_ramp --competition_stage learn
```

训练完整竞赛混合课程：

```bash
./isaaclab_migration/run.sh --terrain robocon --compact_courses \
  --competition_stage learn --num_envs 128 --headless \
  --save_interval 300 --log_dir /home/qyw/WMP/isaaclab_migration/logs/robocon_learn
```

只生成地形预览：

```bash
./isaaclab_migration/preview_terrains.sh --level 14 \
  --types plum_stones_flat plum_stones tire_array high_wall platform ramp platform_gap \
  stair_gap stair_gap_ramp
```

预览器的 `--level` 上限按梅花桩配置为14；其它竞赛地形只有L0–L9。选择具体类型时，预览器按该类型实际有效等级生成。

## 6. 来源与代码位置

- 训练入口参数和 profile 选择：`isaaclab_migration/scripts/train.py`
- 竞赛地形类型、尺寸、课程等级和混合列数：`isaaclab_migration/wmp_lab/competition.py`
- 通用地形生成与课程难度：`isaaclab_migration/wmp_lab/terrain.py`
- 任务成功/失败、接触覆盖与课程更新：`isaaclab_migration/wmp_lab/competition_tasks.py`
- 竞赛细节及梅花桩课程：`isaaclab_migration/OBSTACLE_CURRICULUM.md`
- 楼梯沟壑组合细节：`isaaclab_migration/COMBO_TERRAINS.md`
