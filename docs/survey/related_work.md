# 相关工作调研：图形学与机器人领域如何建模和仿真植物（2026-11）

> 调研方式：4 个并行线程（植物几何建模 / 植物动力学仿真 / 杆与壳求解器及逆静力学 / 2023–2026 最新工作），
> 尽量读一手来源（ACM DL、arXiv、作者主页、Helios 源码文档）。
> 标记：**[读]** 读过方法与局限章节；**[摘]** 只看了摘要或片段；**[待核]** 未能核实。
> 结论里标“推断”的是我们的判断，不是论文原话。

## 0. 一句话结论

“**深树直接求解的杆骨架 + 双向耦合的可形变叶片薄壳 + 让输入形状成为重力平衡态的逆静力学 + 带果实的作物 + 机器人接触**”这个组合，
没有任何一篇现有工作同时做到。最接近的单篇是 Zhao & Barbič 2013（SIGGRAPH），它自己列出的局限
（惯量近似、假设树浅、没有重力预加载、没有接触、叶片刚性）几乎一一对应我们的目标贡献。
但几个组件都有很强的先例，而且有三个团队在快速逼近（见第 6 节）。

---

## 1. 图形学怎么建模植物几何

| 路线 | 代表工作 | 输出 | 对仿真的价值 |
|---|---|---|---|
| 规则 / 文法 | L-system（Prusinkiewicz & Lindenmayer 1990）；Weber & Penn 1995；Xfrog（Lintermann & Deussen 1999）；SpeedTree | 拓扑、节间长度和半径、器官语义（商业工具常烘焙成网格，丢失图结构） | 高，天然带结构 |
| 自组织生长 | Space colonization（Runions 2007）；Self-organizing trees（Pałubicki, SIGGRAPH 2009）；Plastic Trees（Pirk 2012）；Synthetic Silviculture（Makowski 2019）；Rhizomorph（Li 2023）；Interactive Invigoration（Li 2024，体积化的枝条结构） | 树图、半径、年龄；叶片多为实例化模板 | 高，但几乎都是乔木 |
| 反向程序化建模 | Stava 2014（CGF）；Guo 2020（TOG，从图像推断 L-system）；**CropCraft**（arXiv 2411.09693，作物冠层）；FloraForge（2025/26，LLM 写生成器）；**Helios VLM**（arXiv 2603.22622，从图像生成 Helios XML） | 程序参数，继承正向模型的全部结构 | 高；作物方向的工作目标是表型和光照，不是力学 |
| 从图像或点云重建 | Quan 2006（草本！）；Tan 2007/2008；Livny 2010；Li 2021（SIGGRAPH Asia，单图）；Tree-D Fusion（ECCV 2024，60 万棵树）；Neural Hierarchical Decomposition（CVPR 2025）；GaussianPlant（2025）；**LeafFit**（EG 2026，3DGS 转叶片网格资产） | 外观 + 粗结构；结构好的都是“退回到一个程序化模型” | 中；都捕捉到受重力后的形状，没有一篇恢复原始形状 |
| 生成模型（2022–26） | 自回归树生成（SIGGRAPH Asia 2025）；ICLR 2026 图生长；Infinigen | 树图 + 半径 | 低；只有树，没有器官语义 |
| 叶片 | 叶脉（Runions 2005）；叶形发育（Runions 2017）；干枯卷曲（2013）；**萎蔫仿真**（Maggioli, SIGGRAPH Asia 2023，PBD） | 叶片几乎都是模板曲面 | 具备力学属性的叶片模型很少 |

**要点**
- 适合作为可仿真资产来源的，是**能输出显式器官图的程序化或发育模型**（L-system / phytomer / 自组织）。Helios 属于这一类，而且是唯一大量覆盖**带果实草本作物**的。
- 图形学表示普遍缺少：材料参数、**未受力的原始形状**、器官连接与脱落区、**果实**（几乎完全缺席）、作为力学对象的叶片、碰撞几何。
- 图形学文献压倒性地关注乔木；草本加果实的作物只出现在农学方向的工作里（CropCraft、FloraForge、Helios）。

## 2. 图形学怎么仿真植物动力学

### 2.1 最接近的先例：Barbič 组（USC）[读]

**Zhao & Barbič, "Interactive Authoring of Simulation-Ready Plants", TOG 32(4):84, SIGGRAPH 2013**
— <https://viterbi-web.usc.edu/~jbarbic/botanical/>
- 输入多边形网格（Xfrog / SpeedTree），用户把三角形分成可形变域 F、刚性细枝 R1、刚性叶片和果实 R2；每个域单独体素化成六面体 FEM 网格；用户辅助用最小生成树去掉环。
- 运行时是 **Barbič & Zhao 2011 子结构方法**：每个域一个降阶非线性 StVK 有限元模型（5–20 个自由度），父子域运动学耦合，一次从根到叶的半隐式 Newmark 积分。
- 材料：默认统一 E = 1e6 Pa，再整体缩放让根的最低频率为 1 Hz，每层刚度乘 0.1^深度。原文说植物参数“通常拿不到”。
- 规模：桃树 237 根枝、18,536 片叶、330 个果实，每帧 5 ms。
- 叶片通常是刚体公告板；果实是刚体，按界面力阈值脱落。
- **重力：把输入当作未受力形状，所以会下垂（原文图 13）。他们建议用优化做预加载但没有实现，并报告牛顿静平衡求解“条件数很高，导致不稳定”。**
- 局限：没有连续接触、没有环、材料靠手调、没有自碰撞。

**Barbič & Zhao, "Real-time Large-deformation Substructuring", TOG 30(4), SIGGRAPH 2011** [读]
- 明确假设域树“很浅”，子树惯量集中在界面上（只计重量，不计转动惯量）。
- 和完整 FEM 比，最低频率高约 2 倍；更好的那个变体有时不稳定。精确的 Featherstone 式耦合留作未来工作。
- **这正是我们遇到的“深树 + 末端重载”问题，而他们选的近似恰好在这个区间失效。**

**Wang, Zhao & Barbič, "Botanical Materials Based on Biomechanics", TOG 36(4):135, SIGGRAPH 2017** [读]
- 用生物力学幂律（长度、直径、频率）给每个域分配 E 和 ρ；叶片明确做成可形变的；可以从视频（手工跟踪点）拟合参数。
- 这是图形学里给植物分配材料的主要依据；局限是只调最低频率，只设 E 和 ρ。

### 2.2 其他树木动力学

| 工作 | 表示与求解 | 叶片 | 重力 / 原始形状 | 备注 |
|---|---|---|---|---|
| Windy Trees（Pirk, SIGGRAPH Asia 2014）[读] | 刚性边链 + 角向恢复力，显式欧拉 | 叶簇只提供阻力 | 原文不提重力；原始形状塑性漂移 | 双向耦合 SPH 风，可断枝；追求“看起来对”，不是力学 |
| Real-Time Interactive Tree Animation（Quigley … Fedkiw, TVCG 2018）[摘] | 关节刚体 + 可任意硬的弹簧，线性时间 | 叶片和果实可断（博士论文） | 未知 | 单核实时 1 万根枝；正文付费墙未读 |
| Physically Guided Animation of Trees（Habel, EG 2009）[读] | 解耦的梁振子 + 着色器 | 运动纹理 | 不建模 | 线性，只有风 |
| Stam 1997 / Diener 2009 / Shinya 1992 / Weber 2008 / Selino 2013 | 模态或随机方法 | — | — | 风驱动的视觉动画 |
| **Oriented Strands（Hadap, SCA 2006）**[摘] | 分叉的降阶坐标链，**递归、线性时间、全隐式** | 有植被演示 | — | 图形学里最直接的“刚硬树结构直接求解”先例 |
| 攀援植物（Hädrich, EG 2017）[读] | 带方向的粒子，PBD，刚度随年龄混合（非物理） | 每片叶一个粒子 | 重力作为生长和断裂因素 | |
| **Stressful Tree Modeling（Li … Benes, SIGGRAPH 2025）**[读] | 枝条拆成纤维束，Cosserat 杆，**XPBD 迭代 25 次** | 叶片有质量和脱落应变 | 部分折断后下垂 | 实时折断、修剪、风；只做树，叶片不是薄壳 |

### 2.3 机器人领域的植物接触仿真

| 工作 | 表示 / 引擎 | 叶片和果实 | 重力 | 机器人 |
|---|---|---|---|---|
| **Gazebo Plants**（Deng … Michels, arXiv 2402.02570）[读] | PBD Cosserat 杆 | 叶片也是杆图；果实是球，按应变阈值脱落 | **明确局限：会下垂，只能靠加大刚度，原文说应该优化原始 Darboux 向量** | 采摘樱桃番茄，和真实机器人做过标定 |
| **OrchardBench**（Munn, arXiv 2607.06337, 2026-07）[读] | Newton / MuJoCo-Warp，胶囊链接 + EI/ℓ 关节弹簧，GPU 批量 | **叶片刚性、无质量、不参与碰撞**；苹果用系绳挂着，14–23 N 脱落 | 生成的骨架就是无应力状态 + 运动学下垂，**没有逆静力学** | Franka 采摘；需要加“稳定护栏”；代码已开源 |
| **Helios `writePlantStructureUSD`**（v1.3.71–1.3.95, 2026-04 至 11）[读，源码文档] | Isaac Sim 胶囊 + D6 关节，K = EI/L，**全局只有一个 E**，加人工惯量 | 叶片、花、果实是刚体，弹簧挂接 | **原始形状 = 导出时的形状（Helios 已经烘焙了叶柄和叶片下垂），所以重力被算两次** | 没有基准、没有论文；2026 年修过多次稳定性 bug |
| Find the Fruit（arXiv 2505.16547）[读] | Isaac Lab PhysX FEM，单一通用人造植物 | 叶片是 FEM 实体 | 未讨论 | 强化学习拨叶，零样本迁移 |
| Learning to Simulate Tree-Branch Dynamics（Jacob, RA-L 2024）[读] | 刚性链接 + 扭簧，Isaac Gym | — | — | 用 SVGD 从探测数据推断刚度和阻尼 |
| Towards Robotic Tree Manipulation（Kim … Kroemer, 2023）[读] | 刚性链接 + 弹簧阻尼 + GNN | — | — | |
| Safe Leaf Manipulation（Yao … Hauser, ICRA 2025）[摘] | 感知得到的形变图 | 人造甜椒 | — | 低损伤拨叶 |
| Modeling Branches…（Rijal … Gu, IROS 2026）[摘] | 点云到四面体 FEM，迭代估计参数 | — | — | 单根枝条的 Real2Sim |
| Agri-Sim / AgriGen / Push-or-Avoid / 草莓强化学习（IROS 2026）[摘] | 刚体或场景级 | — | — | 系统或场景层面，没有植物力学 |

### 2.4 从视觉得到植物动力学（2024–26）[摘]
- PhysDreamer（ECCV 2024，Gaussian 上的 MPM，演示多为花）；DynamicTree（CVPR 2026，模态运动 + 8,786 个动画树）；Wind on Trees（ECCV 2026 workshop，**负面结果：阻尼完全恢复不了**）。
- 都没有接触、没有机器人，也不是为作物做的。

## 3. 求解器：深树、高刚度、大质量比

| 方法 | 核心 | 树结构 | 刚度 / 质量比 | 局限 |
|---|---|---|---|---|
| **Deul, Kugelstadt, Weiler, Bender, "Direct Position-Based Solver for Stiff Rods", CGF 37(6) 2018** [读全文] | XPBD 时间离散 + 牛顿迭代，KKT 系统用 **Baraff 1996 线性时间树求解器**直接解；有环或接触时和 GS 交替 | 完全支持（演示了 3–5 万段的白桦、柳树、松树，每片叶一个刚体段） | 达到相同精度比 XPBD 快约 100 倍；**质量比没有专门评估** | 只有 CPU，约 1 s/步；叶片是刚体；**忽略了几何刚度**；没有逆静力学 |
| Baraff 1996（SIGGRAPH）| 无环约束的 O(n) 直接解；额外 k 个约束 O(kn)（Schur 补） | 是 | — | 是挂接叶片和接触的模板 |
| Kugelstadt & Schömer 2016（SCA）| 位置 + 朝向的 Cosserat 杆，XPBD-GS | 是 | 差（Deul 测得在树上要 10 万次以上迭代） | **就是我们现在的模型** |
| **Stable Cosserat Rods（Hsu … Yuksel, SIGGRAPH 2025）**[读] | 朝向用闭式准静态 GS，位置用 VBD | 是（演示 3 棵树） | 刚树上 XPBD 完全不收敛；展示的最大质量比只有 46 | 自己承认长链是“最坏情况”；**没有和 Deul 2018 比** |
| VBD（SIGGRAPH 2024）/ **AVBD（Giles … Yuksel, SIGGRAPH 2025）**[读] | 原始空间块下降；AVBD 加增广拉格朗日 | — | AVBD 展示了 5 万比 1 的质量比链 | “信息传播受迭代次数限制”；**未来工作写的是“和全局求解器结合”** |
| XPBD / Small Steps（Macklin 2016/2019）| 对偶空间 GS | — | 需要的子步数取决于质量比，无法预测 | 和我们观察到的一致 |
| Primal/Dual Descent（Macklin, SCA 2020）| 理论框架 | — | 原始法怕刚度比，对偶法怕质量比 | **我们的失效正好是两者同时出现** |
| Geometric Stiffness（Andrews 2017；Tournier 2015）| 加入几何刚度项 | — | 能修复对偶法的质量比问题，“但需要直接求解器” | 在树上直接求解很便宜，和 Deul 天然互补 |
| Multi-layer XPBD（SCA 2024）/ MGPBD（SIGGRAPH 2025）| 由粗到细 / 多重网格 | — | MGPBD 在极端刚度和质量比下不稳定 | |
| Projective Dynamics（2014）/ PD-Cosserat（Soler 2018）| 局部-全局，预分解 | 是 | 刚度高时收敛慢 | |
| DER（Bergou 2008/2010）| 中心线 + 扭转，隐式牛顿 | 分叉处要额外刚体 | 好，但慢 | |
| Super-Helices（Bertails 2006/2009）| 降阶分段螺旋，线性时间 | 只能单根 | 很好 | |
| Featherstone ABA / Quigley 2018 | 降阶坐标，O(n) | 是 | 质量比不是问题 | 刚度高的弹簧要隐式积分；加接触和环很难；**人工惯量（armature）是 hack** |
| IPC / Codim-IPC / Unified Newton Barrier | 牛顿 + barrier | — | 好 | 贵，可作 ground truth |

**杆与叶片薄壳的耦合：没有找到专门针对植物的工作**（Codim-IPC 是一体化但很贵；Umetani 2014 的 PBD 杆端挂接；Baraff 的 O(kn) Schur 补）。

**一个结构性观察（推断）：** 只在叶柄一个点挂接的叶片不会破坏图的无环性。叶片的柔度和惯量可以压缩成挂接点上的等效 6×6 关系，全局仍然是树，线性时间直接解依然成立。只有叶片之间、叶片和茎之间的接触才会引入环。

## 4. 逆静力学（去掉重力下垂）

| 工作 | 内容 | 是否适用于植物 |
|---|---|---|
| Twigg & Kačić-Alesić, "Optimization for Sag-Free Simulations"（SCA 2011）| 优化原始长度等参数，让建模形状就是重力下的平衡态 | 通用 |
| Derouet-Jourdan 2010（TOG）/ 2013（SIGGRAPH Asia，带摩擦接触的逆向头发）| 输入是带接触的静平衡；问题欠定，用均匀参数正则化 | 头发 |
| Bertails-Descoubes 2018（Proc. R. Soc. A）| 一端固定的悬挂 Kirchhoff 杆：**自然形状唯一**，三个线性初值问题 | 单根杆 |
| Hsu 2022（SIGGRAPH，通用两阶段 sag-free）/ **Hsu 2023（SIGGRAPH，sag-free 头发，修正 Darboux 向量）** | 全局线性求平衡力，再局部恢复原始形状 | 头发、布料、FEM |
| Takahashi & Batty 2024/25（arXiv，DER 原始形状优化；active-set Cholesky）| 让净力精确为零，可同时估刚度 | 单根，没提分叉 |
| Chen 2014（SIGGRAPH，反向弹性形状设计）| 渐近数值法 | 体积制造 |
| **植物专用的逆静力学** | **没有找到。** Zhao & Barbič 2013 提出过但没做（牛顿法不稳定）；Gazebo Plants 列为局限 | — |

**关键观察（推断，可能是贡献的核心）：** 根部固定、没有接触的树是**静定结构**。每段的内力等于其远端所有重力载荷（茎、叶、10 g 以上的果实）之和及其力矩，**一次从叶到根的 O(n) 扫描就能精确算出**，再用本构关系的逆在局部求出原始曲率。这把 Bertails-Descoubes 2018 的唯一性结论推广到了带集中载荷的分叉结构。真正的难点（也就是可能的贡献）：
1. 要和离散的正向求解器精确一致（输入必须是**模拟器自己**会收敛到的平衡态）。
2. 叶片薄壳内部是超静定的，需要类似 Hsu 2022 的全局求解，再通过叶柄内力耦合回来。
3. 接触和环需要类似 Derouet-Jourdan 的带摩擦逆问题。
4. **稳定性：** 直立的茎承受压力，可能屈曲。恢复出的原始形状必须让 Hessian 正定。头发类工作都是悬挂问题，从来没碰到过这一点。
5. 合理性约束（原始曲率的范围，以及和刚度一起估计）。

## 5. 断裂和脱落
- Zhao & Barbič 2013（预先指定、按界面力阈值）；Stressful Trees 2025（纤维束断裂）；Gazebo Plants（应变阈值）；OrchardBench（断裂模量 + 果实拉力阈值）。
- 都是阈值模型。**在直接求解器里，脱落就是删掉一棵子树，消元顺序可以复用**（推断），实现起来很便宜。

## 6. 差距矩阵与竞争态势

| | 作物（草本 + 果实） | 可形变叶片 | 深树重载稳定求解 | 逆静力学 | 机器人接触 | 资产库 |
|---|---|---|---|---|---|---|
| Zhao & Barbič 2013 / 2011 / 2017 | 部分（有花和桃树） | 2017 是 FEM 实体 | ✗（惯量集中，假设浅树） | ✗（提出过但失败） | ✗ | 100+ 个手工制作的 |
| Deul 2018 | ✗ | ✗（刚体） | ✓（CPU） | ✗ | ✗ | ✗ |
| Stable Cosserat Rods / AVBD 2025 | ✗ | ✗ | 部分（自认长链是弱点） | ✗ | ✗ | ✗ |
| Stressful Trees 2025 | ✗ | ✗ | XPBD | ✗ | 用户拖拽 | ✗ |
| Gazebo Plants 2024 | ✓（樱桃番茄） | ✗（杆图） | PBD（会下垂） | ✗（列为局限） | ✓ | ✗ |
| **OrchardBench 2026** | ✗（只有苹果） | ✗（刚性、无质量） | 迭代 + 稳定护栏 | ✗ | ✓ | 程序化苹果树 |
| **Helios USD 导出 2026** | ✓（约 30 种） | ✗（刚体） | PhysX + 人工惯量 | ✗（重力算两次） | 依赖 Isaac | ✓（但没有验证） |
| Find the Fruit 2025 | 通用假植物 | ✓（FEM 实体） | — | ✗ | ✓ | ✗ |
| **我们的目标** | ✓ | ✓（薄壳） | ✓（树结构直接求解 + 几何刚度） | ✓（O(n) 静定扫描 + 叶片耦合 + 稳定性） | ✓ | ✓ |

**正在逼近同一方向的团队**
1. **Bailey 组（UC Davis, Helios）：风险最高。** 论文第 4.4 节明确计划做“基于柔性杆弯曲方程的枝条形变”，动机是重果负载和机器人采摘；同时几乎每周都在更新 Isaac Sim 导出功能。建议考虑合作，而不是和他们赛跑。
2. **Munn（OrchardBench）：** 未来工作写了“扩展到其他作物和整形树”，会直接进入我们的空间。
3. **Benes / Pirk / Pałubicki / Michels 组：** 每年 1–2 篇 SIGGRAPH 树木物理论文（纤维束、断裂、萎蔫、风），而且 Gazebo Plants 也有 Michels 参与；下一步很可能做交互或机器人树木仿真。
4. Hsu / Yuksel 组（犹他）：杆求解器和 sag-free 技术的拥有者；如果他们把 Stable Cosserat Rods 扩展到植物，会和求解器贡献重叠。

## 7. 对 SIGGRAPH 投稿的启示

**站得住的贡献组合（按重要性）**
1. **面向植物的混合求解器：** 杆骨架用树结构直接求解（Deul/Baraff 式，加几何刚度修复质量比问题，并且 GPU 并行、多株批量）；单点挂接的叶片薄壳压缩到挂接点（保持树结构）；只有接触和环交给外层迭代（AVBD 式增广拉格朗日，或 Baraff 的 O(kn)）。
   - 要回答审稿人的“和 Deul 2018 有什么不同”：可形变薄壳叶片、几何刚度带来的质量比鲁棒性、GPU 加批量、作物级规模和接触。
2. **植物逆静力学：** O(n) 静定内力扫描 + 局部恢复原始 Darboux 向量（参考 Hsu 2023 的四元数修正）+ 叶片薄壳耦合 + **屈曲稳定性保证** + 和离散正向求解器精确一致。
   - 要回答“为什么不直接用 Hsu 2022/2023”：树结构让问题精确且线性时间；直立茎受压会失稳，而悬挂的头发永远不会遇到这个问题。
3. **应用：** Helios 作物资产库（8 种以上）+ 机器人任务（拨叶、穿行、采摘时果柄断裂 = 删除子树）。

**审稿人会期待的基线**
- 正向求解：XPBD + 子步（我们现在的）；**Deul 2018**（Bender 的 PositionBasedDynamics 开源库里可能有实现，待核）；VBD、**AVBD**；**Stable Cosserat Rods**；DER 隐式牛顿；PD-Cosserat；降阶坐标关节树（MuJoCo / Newton，含人工惯量消融，也就是 OrchardBench 和 Helios USD 的做法）；完全收敛的牛顿或 IPC 作为 ground truth。
- 压力测试：深度（30 段以上、1–2 m）、质量比（果实比果柄 1e3–1e5）、刚度比（木质部比叶片）。
- 逆静力学：Twigg 2011、Hsu 2022、Hsu 2023、Takahashi & Batty、Derouet-Jourdan 2013；指标是输入姿态随时间的漂移。
- 植物：Zhao & Barbič 2013 + 2017、Deul 2018 的树、Stressful Trees 2025、OrchardBench 2026。

## 8. 待核实
- Quigley 2018 的方法细节（付费墙）；Deul 求解器的开源实现是否支持完整的树；MuJoCo / PhysX 人工惯量的细节；Andrews 2017、Umetani 2014、Ly 2018 的准确出处。
- Agri-Sim、Push-or-Avoid、草莓强化学习（IROS 2026）的植物模型（只看了摘要）。
- SIGGRAPH Asia 2026 的论文列表还只是有条件录用。
- 尚未系统检索 ICRA/IROS 2026 论文集和 CVPR/ICCV 2025–26。

## 主要链接
- Zhao & Barbič 2013 <https://viterbi-web.usc.edu/~jbarbic/botanical/> · Barbič & Zhao 2011 <https://viterbi-web.usc.edu/~jbarbic/substructuring/> · Wang 2017 <https://viterbi-web.usc.edu/~jbarbic/proceduralMaterials/>
- Deul 2018 <https://animation.rwth-aachen.de/media/papers/2018-CGF-Rods.pdf> · Stable Cosserat Rods <https://jerryhsu.io/wp-content/uploads/2025/05/Sig25__Stable_Cosserat_Rods.pdf> · AVBD doi:10.1145/3731195
- Sag-free 2022 <https://graphics.cs.utah.edu/research/projects/sag-free-simulations/> · Sag-free hair 2023 doi:10.1145/3592143 · Takahashi & Batty <https://arxiv.org/abs/2409.12362>, <https://arxiv.org/abs/2412.16461> · Derouet-Jourdan 2013 <https://hal.archives-ouvertes.fr/hal-00857559>
- Windy Trees <https://www.cs.purdue.edu/homes/bbenes/papers/Pirk14ToG.pdf> · Stressful Trees <https://www.cs.purdue.edu/cgvlab/www/publications/Li25Sigg> · ADONIS <https://www.cs.columbia.edu/cg/adonis>
- Gazebo Plants <https://arxiv.org/abs/2402.02570> · OrchardBench <https://arxiv.org/abs/2607.06337> （代码 <https://github.com/humphreymunn/orchardbench>）· Find the Fruit <https://arxiv.org/abs/2505.16547> · Tree-branch dynamics <https://arxiv.org/abs/2306.03410>
- Helios 框架 <https://arxiv.org/abs/2512.17966> · Helios VLM <https://arxiv.org/abs/2603.22622> · CropCraft <https://arxiv.org/abs/2411.09693> · FloraForge <https://arxiv.org/abs/2512.11925> · LeafFit <https://cgenglab.github.io/en/publication/eg26_leaffit> · GaussianPlant <https://arxiv.org/abs/2512.14087> · Tree-D Fusion <https://arxiv.org/abs/2407.10330>
- Pałubicki 2009 <https://www.algorithmicbotany.org/papers/selforg.sig2009.html> · Synthetic Silviculture <https://storage.googleapis.com/pirk.io/projects/synthetic_silviculture/index.html> · 叶脉 2005 <https://www.algorithmicbotany.org/papers/venation.sig2005.html> · 萎蔫 2023 <https://gladia.di.uniroma1.it/publication/maggioli-2023-plant/>
