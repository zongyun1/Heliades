# 公开作物数据调研：结构、图像性状、力学参数（2026-11）

用途：智能体改写 Helios 生成器，使生成植株匹配真实植株。数据有三种用法：
(a) 结构目标；(b) 留出测试集；(c) 资产的物理参数标注。

核实标记：**OK** 表示调研时打开过页面或仓库；**部分** 表示只见到搜索摘要、镜像或论文；**未核实** 表示没能确认。
数值标记：**[V]** 表示在原文或摘要中看到；**[S]** 表示来自搜索摘要或二手引用，写入资产元数据前必须回原文核对；**[U]** 表示未找到来源。

---

## 0. 结论

- **第一个物种选番茄。** 唯一同时具备人工骨架、人工结构性状、照片加性状、生育期标签的作物。
  - 目标集：**TomatoWUR**。44 株，人工测量的节间长、茎粗、叶角、叶序角，加人工骨架。
  - 留出测试：**TomatoPGT**。生长时间序列，骨架图。
  - 照片加性状：**AGC 第 4 届预试验**。479 株，株高、叶面积、红果数。
  - 照片条件与生育期：**TomatoMAP**。6.4 万张多视角照片，BBCH 50 类，器官检测框。
  - 生长动态：**Pheno4D**。
- **第二梯队**：
  - 玉米：MaizeField3D、Single Maize、Miao 叶片计数、UNL 时间序列。
  - 草莓：LAST-Straw。
  - 甜菜：BonnBeetClouds3D（CC0）。
  - 高粱：TERRA-REF。
  - 葡萄：3D2cut 结构图，但只有休眠期。
  - 菜豆、豇豆：Galba PlantEye。
- **缺口**：
  - 辣椒、大豆、水稻、樱桃番茄、生菜（结构方面）几乎没有带结构真值的数据。
  - 茎粗、叶柄插入角、果实位置：除番茄和草莓外几乎没人测。
  - 没有任何数据集配文字描述，文字需要从元数据（品种、生育期、性状）生成。这本身是我们数据集的一个贡献。
- **力学参数**：
  - 覆盖较好：谷物茎秆模量、整株频率和阻尼、采摘断裂力。
  - 番茄、辣椒、豆类的茎和叶柄 E/G、双子叶中脉 EI、草本鲜茎密度基本空白。
  - 没有任何公开的作物力学数据集。系统整理带出处的物理参数，可以作为数据集的贡献。
- **许可**：
  - Pheno4D、LAST-Straw、TomatoPGT、Single Maize 未声明数据许可。
  - MaizeField3D 和 Laboro Tomato 限非商用。
  - 发布衍生数据前必须确认。

---

## 1. 3D 结构数据（点云 / 多视角 / 骨架）

| 数据集 | 作物 | 模态 | 规模 / 生育期 | 标注 | 许可 | 链接 | 论文 | 相关度 → 指标 | 核实 |
|---|---|---|---|---|---|---|---|---|---|
| **TomatoWUR** | 番茄 | 15 相机轮廓重建加 RGB | 44 株 | 语义和实例分割；**人工骨架（CSV）**；**人工测量**节间长、节间直径、叶角、叶序角 | CC BY-SA 4.0 | data.4tu.nl/datasets/e2c59841-4653-45de-a75e-4994b2766a2f ；github.com/WUR-ABE/TomatoWUR | Data in Brief 61:111852, 2025 | **高**：节间、茎粗、叶/叶柄角、叶序、拓扑、株高 | OK |
| **TomatoPGT** | 番茄（3 品种） | 多视角 MVS，约 67 张/次 | 3 株共 42 次扫描，营养生长前中期时间序列 | 语义（主茎、节点、复叶、叶柄、侧枝）和实例；**骨架图 JSON**；性状由几何计算（非人工） | 未声明（论文 CC BY） | data.mendeley.com/datasets/72md54c7n7/1 | Data in Brief 2026 | **高**：株高、节间、插入角、叶序、生长 | OK |
| **Pheno4D** | 玉米 7×12 天，番茄 7×20 天 | 激光（Romer 臂） | 224 片点云，标注 49 片玉米、77 片番茄 | 土/茎/叶，叶实例**跨天一致** | 未声明 | ipb.uni-bonn.de/data/pheno4d ；HF Voxel51/pheno4d | PLoS ONE 2021 | **高**：叶数、叶长/面积、株高、生长动态 | OK |
| 4D Plant Registration | 玉米、番茄各 3 株 | 激光 | 10 天逐日 | 是否含骨架未核实 | 未声明 | ipb.uni-bonn.de/data/4d-plant-registration | Magistri et al. IROS 2020 | 低–中 | OK |
| **MaizeField3D** | 玉米（大田多样性群体） | 地面激光 | 1045 片，其中 520 片已分割 | 叶实例按位置排序、茎秆；**NURBS 叶片拟合**；元数据（叶数、雄穗、果穗） | **CC BY-NC 4.0** | huggingface.co/datasets/BGLab/MaizeField3D | Plant Phenomics 2026；arXiv 2503.07813 | **高**：叶数、长宽、曲率、面积、群体分布 | OK |
| Single Maize Point Cloud | 玉米 5 品种 | 激光 | 428 株，2–12 叶 | 茎、叶实例（自下而上编号） | 未声明 | github.com/syau-miao/SignleMaizePointCloudDataSet | ISPRS J. P&RS 2024 | **高**：叶数、叶长/面积、出叶顺序 | OK |
| Crops3D | 玉米、水稻、小麦、番茄、马铃薯、棉花、甘蓝、油菜 | TLS、结构光、MVS 混合 | 1230 | 器官（土/茎/叶/果），株实例；无叶实例 | CC BY 4.0（镜像所述） | github.com/clawCa/Crops3D ；figshare 10.6084/m9.figshare.27313272 | Sci. Data 2024 | 中 | OK |
| Soybean-MVS | 大豆 5 品种 | MVS | 102 个模型，13 个生育期 | 叶、主茎、茎 | 未核实 | **下载地址未找到** | Agriculture 13:1321, 2023 | 中 | 未核实 |
| Legume PlantEye（Galba） | 菜豆 50、豇豆 46、利马豆 58、绿豆 102 | PlantEye 激光三角测量加多光谱 | 256 盘，幼苗 | 子叶、叶、叶柄、茎；CSV | CC BY 4.0 | github.com/kit-pef-czu-cz/3d-point-cloud-dataset-plants | Sci. Data 2025 | 中–高（豆类唯一）；只有顶视，遮挡重 | OK |
| PLANesT-3D | 辣椒 10、月季、茶藨子 | SfM/MVS，有尺度 | 34 株 | 叶/茎，叶实例 | ? | github.com/visionlab-ogu/PLANesT-3D | arXiv 2407.21150 | 中：辣椒叶数/大小 | OK |
| Pepper-4D | 辣椒 | 高精度点云 | 29 株共 916 片，时空序列 | 器官分割、新器官检测 | 未核实 | doi 10.3390/plants15040599 | Plants 15(4):599, 2026 | 可能高，**细节和下载未核实** | 部分 |
| BUP-ST20 | 甜椒（温室） | 机器人 RGB-D | 时空序列 | 弱标注（果、果柄跟踪） | CC BY 4.0 | bonndata 10.60507/FK2/NUMVO1 | – | 低–中：果数/位置 | 部分 |
| **LAST-Straw** | 草莓 2 品种 | 手持结构光 | 6 株 × 14 次，共 11 周 | 28/84 已逐点标注（叶、茎、花、果、冠）；**茎骨架真值** | 未声明 | lcas.github.io/LAST-Straw | ECCV-W CVPPA 2024；arXiv 2403.00566 | **高**（草莓）：叶柄长/角、小叶、果数/位置 | OK |
| **BonnBeetClouds3D** | 甜菜 48 品种 | 无人机摄影测量 | 186 株，2661 片叶 | 株/叶实例，叶尖/叶基关键点；**专家测量叶长、叶宽、叶柄长** | **CC0** | bonnbeetclouds3d.ipb.uni-bonn.de | IROS 2024 | **高**（甜菜） | OK |
| Wheat3D PartNet | 小麦 3 品种 | MVS | 1303 株 | 穗/非穗 | ? | figshare.edgehill.ac.uk …/30228148 | COMPAG 238:110697, 2025 | 低–中：株高、穗 | 部分 |
| MMR / MMW | 水稻、小麦 | NeRF 多视角加点云 | ? | 籽粒、叶、茎实例 | ? | **仓库未找到** | Plant Phenomics 2026 | 中（若发布） | 未核实 |
| Plant3D（Navlakha） | 番茄、烟草、高粱、拟南芥 | 激光（FaroArm） | 559 个枝条，每个 20–30 个时间点，5 种处理 | 原始点云 | CC BY 4.0 | data.mendeley.com/datasets/9k7zctdyhs/1 ；Dryad 10.6075/J0DZ06NT | Conn 2017；Ziamtsov & Navlakha 2019 | 中：分枝、生长（骨架要自己提取） | OK |
| Sorghum LiDAR | 高粱 | LiDAR | ? | 叶、茎、穗 | CC BY 4.0 | data.mendeley.com/datasets/pfnfzrmrg7/1 | – | 中 | OK |
| TLS-Grapevine2024 | 葡萄 2 品种 | TLS 加无人机 | 52 株 × 9–10 个 BBCH 期 | 无逐点标注；LAI、果穗、产量、修剪量真值 | CC BY 4.0 | zenodo.org/records/16751663 | – | 中 | OK |
| CMU/Cornell 冬季葡萄 | 葡萄（休眠） | 双目点云 | 144 株 | 枝条类别、修剪量 | ? | labs.ri.cmu.edu/aiira/resources/ | arXiv 2307.11706 | 低–中 | OK |
| 生菜鲜重（Bloch） | 生菜 | 4× RealSense | 402 片 | 鲜重 | 开放 | zenodo.org/records/8410252 | Plant Methods 2025 | 低 | 部分 |
| ROSE-X | 月季（非 Helios 作物） | X 射线 CT | 11 株 | 体素器官标注 | CC0/BY | uabox.univ-angers.fr/…/rnPm5EHFK6Xym9t | Plant Methods 2020 | 低（可测遮挡处理） | OK |
| PlantDreamer data | 菜豆 12、羽衣甘蓝、薄荷 | 3DGS 加合成 L-system 网格 | 21 株 | 真实-合成配对 | ? | plantimages.nottingham.ac.uk/datasets.html | ICCV-W CVPPA 2025 | 中（sim-to-real 外观） | OK |

**不是数据集**：3DPhenoMVS（只有方法流程）、Pheno-Deep（2D 计数模型）。Plant-Mesh 未找到。

---

## 2. 图像加性状数据（照片条件生成和评估）

| 数据集 | 作物 | 图像 | 规模 | 标注 / 实测性状 | 许可 | 链接 | 相关度 | 核实 |
|---|---|---|---|---|---|---|---|---|
| **第 4 届 Autonomous Greenhouse Challenge 预试验**（WUR 2024） | 矮生番茄 | 单株 RGB-D，加冠层时间序列 | 479 株单拍 | **株高、鲜重、叶面积、红果数**，每周人工测量 | CC BY 4.0 | data.4tu.nl/datasets/e1ee9de9-6ce9-4502-a37c-34b5b1372bed | **高** | OK |
| 第 4 届 AGC 正式赛 | 矮生番茄 | 冠层 RGB-D，每天 4 张，约 2 个月 | 6 个隔间，约 31 GB | 果数、果重、成熟度、气候 | CC BY 4.0 | data.4tu.nl/datasets/fa102772-32db-4b30-bace-12f2016722ce | 中 | OK |
| **第 3 届 AGC 生菜** | 生菜 4 品种 | 顶视 RGB-D 时间序列 | 388 组 | 真值 JSON（据记忆含鲜重/干重、高度、直径、叶面积，页面未列出） | CC BY 4.0 | doi.org/10.4121/15023088.v1 ；10.4121/21960932.v1 | **高**（生菜） | 部分 |
| **TomatoMAP**（IPK 2025） | 番茄 | 多视角（4 个高度 × 12 个角度），时间序列 | 64464 张 RGB，3616 张带分割 | **BBCH 生育期 50 类**；叶、花序、花/果、侧枝、整株检测框；花果实例掩码（无人工尺寸） | CC BY 4.0 | doi.org/10.5447/ipk/2025/14 ；arXiv 2507.11279；Sci Data 2026 | **高**（生育期、器官计数、多视角） | OK |
| Horti-M3-Tomato（Sci Data 2026） | 番茄（中国东北温室） | RGB 时间序列，2023–25 | ? | 每周人工**株高、茎粗、叶长宽、叶面积、叶数**，产量 | ? | nature.com/articles/s41597-026-07074-w | **高（若可下载）** | 部分：仓库未找到 |
| Hoogstraten 小气候（2018–20） | 番茄（商业温室） | 无图像 | 每周 | 果穗数和穗重、茎长、茎粗、品质 | CC BY 4.0 | data.mendeley.com/datasets/tkbkzdt5nr | 中（结构先验） | OK |
| Laboro Tomato | 番茄、樱桃番茄 | 温室近景 | 804 张 | 果实实例、3 种成熟度 × 大果/樱桃 | CC BY-NC-SA 4.0 | github.com/laboroai/LaboroTomato | 低–中 | OK |
| TomatOD | 番茄 | 温室 | 277 张，2418 个果 | 果检测框 | CC BY-NC-SA | github.com/up2metric/tomatOD | 低 | 部分 |
| StrawDI_Db1 | 草莓 | 田间单视角 | 3100 张 | 果实例 | 引用即可 | strawdi.github.io | 低–中 | 部分 |
| Sweet Pepper & Peduncle / BUP19/20 | 甜椒 | 温室行 | 620 张 | 果、果柄掩码 | CC BY-SA 4.0 | agrobotics.uni-bonn.de/sweet_pepper_dataset | 低 | 部分 |
| **3D2cut Single Guyot** | 葡萄（休眠） | 田间单视角 | 1511 张 | **完整的 2D 植株结构图**：节点、枝型、朝向 | 查看 Zenodo | idiap.ch/en/dataset/3d2cut ；zenodo 7679898 | **高**（葡萄拓扑） | OK |
| Embrapa WGISD | 葡萄 | 田间 | 300 张，4432 个果穗 | 果穗框/掩码 | CC BY-NC 4.0（来源有冲突） | github.com/thsant/wgisd | 低–中 | 部分 |
| wGrapeUNIPD-DL | 白葡萄 | 侧视，多生育期 | 373 张 | 果穗框；部分有田间计数 | 开放 | zenodo 4066729 | 低–中 | 部分 |
| GWHD 2021 | 小麦 | 田间俯视 | 约 6500 张，30 万+ 穗 | 穗框 | CC BY 4.0 | zenodo.org/records/5092309 | 中（穗数） | OK |
| **TERRA-REF** R1 | 高粱（另有小麦、生菜） | 龙门架 RGB/3D/热红外/高光谱 | 2 季，约 300 个基因型 | **60+ 项人工测量**（每小区 CSV，具体列未核实） | Dryad CC0 | datadryad.org/dataset/doi:10.5061/dryad.4b8gtht99 | **高**（谷物） | 部分 |
| Miao 玉米叶片计数 | 玉米 | 温室侧视（LemnaTec） | 4633 张真实加 4060 张模拟 | **叶数** | PDDL | dc.cyverse.org/dataset/f491cd2b-44cd-45a0-bb80-6094830e2171 | **高**（叶数） | OK |
| UNL-CPPD / Panicoid Phenomap-1 / UNL-3DPPD | 玉米、高粱等 | 侧视，逐日时间序列 | CPPD 13 株；Phenomap 176 株 × 27 天 | CPPD 有叶/茎真值；其余只有图像 | 未声明 | plantvision.unl.edu/datasets | 中–高（生长曲线） | OK |
| Artemis / Image Safari（2026） | 菜豆、豇豆、大豆、高粱等 | 田间 | 139 万加 608 万张，已标注 4.05 万张 | 实例/框；农艺、基因型元数据 | CC BY-SA 4.0 | registry.opendata.aws/artemis-image-safari | 中 | OK |
| 菜豆叶片尺寸 | 菜豆 | 单叶 | 612 片叶 | 叶长宽 | ? | Mendeley | 中（叶大小） | 部分 |
| MuST-C（Bonn 2025） | 甜菜、小麦、玉米、大豆、马铃薯、蚕豆 | 无人机/地面车 RGB、多光谱、LiDAR | 多时相 | 人工测量 LAI、生物量 | CC BY 4.0 | ipb.uni-bonn.de/data/MuST-C | 低–中 | OK |
| CropCraft（2024） | 大豆、玉米 | 田间多视角 | ? | 人工叶面积、叶角 | – | arXiv 2411.09693（**数据未公开**） | 中 | 未核实 |
| CVPPP / KOMATSUNA | 拟南芥、烟草 / 小松菜 | 顶视 / 多视角 | 约 800 / 约 1500 | 叶实例、计数 | 研究用 | plant-phenotyping.org | 低（非 Helios 作物，可用于预训练） | OK |

未找到：辣椒整株结构表、水稻侧视加分蘖数、Miao 15 万张全集、Horti-M3-Tomato 仓库、"Pheno-Inspect"。

---

## 3. 力学参数（资产物理标注）

| 作物 | 器官 | 量 | 数值 | 方法 | 出处 | 标记 |
|---|---|---|---|---|---|---|
| 玉米 | 茎皮 | E | 弯曲 10.06±1.51 GPa；压缩 10.15/12.87；拉伸 11.5–12.4；各方法 6–16 GPa | 三点弯、压、拉，干茎（含水 10–15%，鲜茎更低） | Al-Zube et al. 2018, Plant Methods 14:11, doi 10.1186/s13007-018-0279-6 | [V] |
| 玉米 | 茎皮条 | E、强度 | 4–12 GPa；64–197 MPa | 微型三点弯 | Oduntan et al. 2024, Eur. J. Agron. | [S] |
| 玉米 | 整段茎 | 表观 E | 10–674 MPa，随直径增大而降低 | 三点弯，10 个品种 | IJABE（ijabe.org …/8043） | [S] |
| 玉米 | 叶中脉/叶片 | EI | 中脉主导；叶片在叶中部约贡献 50%；柔度从叶基到叶尖近指数增长 | 活体弯曲加复合梁模型 | Moulia, Fournier & Guitard 1994, J. Mater. Sci. 29:2359；Moulia & Fournier 1997 | [S] |
| 玉米 | 整株 | f_n、ζ | 约 1.54 Hz，ζ 约 0.11 | 田间自由振动 | Flesch 1988, Purdue PhD | [S] |
| 玉米、燕麦、油菜 | 整株 | f_n、ζ | 有测量，数值未取得 | 田间位移跟踪 | Joseph et al. 2020, JWEIA 202:104169 | [U] |
| 小麦 | 茎 | E | 4.76–6.58 GPa（随成熟升高） | 四点弯 | O'Dogherty et al. 1995, J. Agric. Eng. Res. | [S] |
| 小麦 | 茎 | E | 0.65–1.82 GPa | 三点弯 | Tavakoli et al. 2009 | [S] |
| 小麦 | 茎（空心/实心） | E、强度 | 0.1–3.5 GPa；4.2–43.9 MPa | 三点弯 | U. Saskatchewan 学位论文 | [S] |
| 小麦 | 单茎 | ζ | 0.08 | 风洞 | Sterling et al. 2003, J. Agric. Sci. | [S] |
| 小麦 | 整株 | 倒伏模型 | 阻尼谐振子，f_n 随季节变化 | 田间标定 | Berry et al. 2003, Field Crops Res.；Baker et al. 2014, J. Theor. Biol. | [S] |
| 谷物 16 品种 | 整株 | f_n | 平均 1.37 Hz | 半球视频 | Plant Methods 2019（PMC6533727） | [S] |
| 水稻 | 秆 | E、EI | 按品种报告，数值未取得 | 三点弯 | Chigira et al. 2020, Sci. Rep.；Ookawa 组 2018, Rice 11 | [U] |
| 高粱 | 茎皮 | 弯曲模量 | 仅相对值（突变体低 22–41%） | 三点弯 | Mengistie et al. 2024, Crops 4(1):2 | [S] |
| 番茄 | 茎 | E、G | 文中表 1，未取得 | 三点弯、剪切 | Agriculture 2024, 14(4):531 | [U] |
| 番茄 | 幼苗 | 拉伸 E | 7.36±2.80 MPa | 拉伸 | JAET (NIAE) | [S] |
| 番茄 | 果柄离层 | 剪切力 | 17.4 N（理论 17.9 N） | 拉、弯、剪 | Weng et al. 2024, Agronomy 14(10):2274 | [S] |
| 樱桃番茄 | 萼/节 | 脱落力 | 0.58–2.46 N，取决于拉的角度 | 拉拔 | 来源未确认 | [U] |
| 加工番茄 | 萼/节/果梗 | 脱落 | 萼约 14.3 N，节约 9.6 N，果梗弯矩约 0.032 N·m | 拉、弯 | J. Jpn. Soc. Farm Work Res. | [S] |
| 草莓 | 果梗 | 脱落力 | 直拉 13.94 N（Ø2.17 mm）；弯拉 3.17 N | 拉拔 | Dimeas et al. 2013 | [S] |
| 草莓 | 果梗 | 切断/夹持 | 峰值切断力 5.8–7.2 N（建议 15 N）；夹持上限 10 N | 刀片切割台 | Rajendran et al. 2022, arXiv:2207.12552 | [V] |
| 辣椒 | 果柄 | 脱柄力 | 从绿熟到红熟先升后降，数值未取得 | 测力计 | Front. Genet. 2023, doi 10.3389/fgene.2023.1114832 | [S] |
| 葡萄 | 果粒柄 | 脱落力 | 多品种测量，数值未取得 | 质构仪 | Forleo et al., OIV 2024；Front. Nutr. 2020 | [S] |
| 葡萄 | 绿枝 | E | 弯曲 5890 MPa；轴向 797；径向 79.8 | 弯曲/压缩 | J. King Saud Univ. Sci. 2024 | [S] |
| 苹果（参考） | 果柄 | 脱落力 | 28.3 N 拉力加 0.257 N·m 扭矩可摘下 95%；田间约 20–25 N | 拉/扭 | MSU 学位论文 etd/50260 | [S] |
| 大豆、菜豆、生菜、甜菜 | 茎/叶柄 | E | **未找到**；鹰嘴豆茎 494 MPa 可作豆类替代值 | – | – | [U]/[S] |
| 通用 | 叶柄 | G/E、EI/GJ | G ≈ E/10（杨树叶柄）；EI/GJ 约 4.3（Vogel 给出 2.7–8） | 弯+扭 | Niklas 1991, Am. J. Bot.；Sci. Rep. 2018, doi 10.1038/s41598-018-34588-0 | [S] |
| 通用 | 叶 | 厚度 | LT ≈ 1/(SLA×LDMC)，残差标准误 0.11，n=1039 | 荟萃分析 | Vile et al. 2005, Ann. Bot. 96:1129 | [V] |
| 通用 | 叶片 | 力学 | 夹层板（stress-skin panel）模型（综述） | – | Niklas 1999, New Phytol. 143:19 | [S] |
| 各作物 | 果实 | 质量 | 樱桃番茄 10–25 g；大番茄 100–300 g+；甜椒 100–250 g；草莓 10–50 g；葡萄粒 1–10 g，果穗 0.1–1 kg | – | 无来源，需按品种查 | [U] |
| 综述 | 风–植物动力学 | f_n、ζ、Cauchy 数 | – | – | de Langre 2008, Annu. Rev. Fluid Mech. 40:141；Gardiner, Berry & Moulia 2016, Plant Sci. 245:94 | [V] |

**数据库**：
- TRY（try-db.org）有 SLA、LDMC、叶厚、茎比密度、株高，需申请访问，以干物质性状为主，无力学参数。鲜重比叶重可以用干 LMA / LDMC 换算，叶厚可以用 Vile 2005 的公式估计。
- Global Spectrum（Sci. Data 2022）中，草本的茎比密度是由 LDMC 填补的，慎用。
- 未找到公开的作物茎模量、阻尼或脱落力数据集。

**缺口（需要自己测量或推断）**：
- 番茄、辣椒、菜豆、大豆茎和叶柄的 E/G。
- 双子叶作物中脉 EI；生菜、甜菜叶柄/叶片 EI。
- 草本鲜茎密度。
- 按品种的果实质量。
- 双子叶作物的阻尼。

可选补法：
- 简单悬臂弯曲测量，用已知砝码。
- 用拨动视频反推刚度，即原 Real2Sim 模块。

---

## 4. 对最小闭环（L0）的数据安排

| 角色 | 数据 | 用到的指标 |
|---|---|---|
| 目标（调参） | TomatoWUR | 节间长、茎粗、叶角、叶序角、骨架拓扑、株高 |
| 留出测试 | TomatoPGT（加上 Pheno4D 番茄） | 同上，加上生长时间序列 |
| 照片条件 | AGC 第 4 届预试验、TomatoMAP | 株高、叶面积、果数、BBCH 生育期、器官计数 |
| 物理标注 | 第 3 节 | 先用文献区间加先验，并标注出处和不确定度 |

下一步：
1. 下载 TomatoWUR，确认骨架 CSV 和人工性状的格式。
2. 从 Helios 番茄算出同样的指标，看看与实测的差距。这一步就是闭环的第一个误差信号。
