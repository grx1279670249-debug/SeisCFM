import numpy as np
# import opsvis as ops
import matplotlib.pyplot as plt
import random
# from scipy.stats import qmc
import os
import math
from tqdm import trange
from openseespy.opensees import *

# os.environ('') = '0'

####################################################################################################
# 单位：
# 长度——m
# 力——kN

####################################################################################################
pi = 3.1415
g = 9.81

'''--------初始化数据集-------'''
############# 参数计算 & 转换 ###############
### girder

ECG = 3.0e7
areaGD = 1.234520E+01
IyGD = 80.71427
IzGD = 2.070309
TGD = 7.170212
Ec30 = 3.0e7
Gc30 = 0.4 * Ec30

### 钢臂参数
# 横隔板，连接主梁的竖向钢臂，桥台

ERigid = 2.0e8
GRigid = 1.0e8
areaRigid1 = 1000
IzRigid1 = 5000
IyRigid1 = 5000
TRigid1 = 30000

### 碰撞参数

# 1 kip/in^(3/2) = 0.0347kN/mm^(3/2)
# 计算中单位为kN/mm^(3/2)
kh_impact = 25000 * 0.0347
n_impact = 1.5
e_impact = 0.65
dm_impact = 10
a_impact = 0.1

# 能量
E_impact = kh_impact * (dm_impact ** (n_impact + 1)) * (1 - (e_impact ** 2)) / (n_impact + 1)
# 等效刚度
Keff_impact = kh_impact * (math.sqrt(dm_impact))
# 屈服位移
dy_impact = a_impact * dm_impact
# k1------>kN/m
# 每侧两个碰撞单元，应除以2
K_impact_1 = (Keff_impact + E_impact / (a_impact * (dm_impact ** 2))) * 1000 / 2
# k2------>kN/m
# 每侧两个碰撞单元，应除以2
K_impact_2 = (Keff_impact - E_impact / ((1 - a_impact) * (dm_impact ** 2))) * 1000 / 2

# 间隙抽样获得

#### 盖梁参数
areaCB = 2.7
IzCB = 0.50625
IyCB = 0.729
TCB = 1.011819
ECB = 3.096270E+07

### 桥台——回填土模型参数
Rf_abutment = 0.7
Gap_abutment = 0.01

# 初始刚度与卸载刚度抽样获得

## 桥台——桩模型参数
pinchx_abutment_pile = 0.75
pinchy_abutment_pile = 0.5
# x1=0.3in
x1a_abutment_pile = 0.0254 * 0.3
# x2=1in
x2a_abutment_pile = 0.0254 * 1

# 墩-盖梁钢臂参数
# rigid frame No.2
areaRigid2 = 500
IzRigid2 = 50000
IyRigid2 = 50000
TRigid2 = 3000

for III in trange(403):

    # remove existing model
    wipe()

    # set modelbuilder
    model('basic', '-ndm', 3, '-ndf', 6)

    '''-------------------------定义线性坐标变换---------------------------------'''
    # column
    geomTransf('Linear', 1, -1, 0, 0)
    # 主梁
    geomTransf('Linear', 2, 0, 1, 0)
    # cap beam
    geomTransf('Linear', 3, 0, 0, -1)

    '''-------------------------supstructures---------------------------------------'''
    Ls = 30
    Ls_step = Ls / 6

    # girder nodes
    for i in range(19):
        node(i + 1, i * Ls_step, 0, 0)

    # 墩1处对应node7
    # 墩2处对应node13

    # girder elements
    for i in range(18):
        element('elasticBeamColumn', i + 1, i + 1, i + 2, areaGD, ECG, Gc30, TGD, IyGD, IzGD, 2)


    '''----------------------------cap beam & Rigid--------------------------------------'''

    # start from node 20
    # start from element 19

    # 支座数量为6
    # 支座间距为4

    # 钢臂1 nodes
    wg_step = 17.37 / 8
    for i in range(9):
        node(20 + i, 6 * Ls_step, wg_step * i - 4 * wg_step, -1)

    # 钢臂1 elements
    element('elasticBeamColumn', 19, 7, 24, areaRigid1, ERigid, GRigid, TRigid1, IyRigid1, IzRigid1, 1)
    for i in range(8):
        element('elasticBeamColumn', 20 + i, 20 + i, 21 + i, areaRigid1, ERigid, GRigid, TRigid1, IyRigid1, IzRigid1, 3)

    # 钢臂2 nodes
    for i in range(9):
        node(29 + i, 12 * Ls_step, wg_step * i - 4 * wg_step, -1)

    # 钢臂2 elements
    element('elasticBeamColumn', 28, 13, 33, areaRigid1, ERigid, GRigid, TRigid1, IyRigid1, IzRigid1, 1)
    for i in range(8):
        element('elasticBeamColumn', 29 + i, 29 + i, 30 + i, areaRigid1, ERigid, GRigid, TRigid1, IyRigid1, IzRigid1, 3)

    # cap beam 1#
    # nodes
    for i in range(9):
        node(38 + i, 6 * Ls_step, wg_step * i - 4 * wg_step, -1)

    # elements
    for i in range(8):
        element('elasticBeamColumn', 37 + i, 38 + i, 39 + i, areaCB, ECB, Gc30, TCB, IyCB, IzCB, 3)

    # cap beam 2#
    # nodes
    for i in range(9):
        node(47 + i, 12 * Ls_step, wg_step * i - 4 * wg_step, -1)

    # elements
    for i in range(8):
        element('elasticBeamColumn', 45 + i, 47 + i, 48 + i, areaCB, ECB, Gc30, TCB, IyCB, IzCB, 3)

    '''----------------------- abutment ------------------------------'''

    # start from node 56
    # start from element 53

    # 梁端L
    # nodes
    for i in range(9):
        node(56 + i, 0, wg_step * i - 4 * wg_step, -1)

    # elements
    element('elasticBeamColumn', 53, 1, 60, areaCB, ECB, Gc30, TCB, IyCB, IzCB, 1)
    for i in range(8):
        element('elasticBeamColumn', 54 + i, 56 + i, 57 + i, areaCB, ECB, Gc30, TCB, IyCB, IzCB, 3)

    # 梁端R
    # nodes
    for i in range(9):
        node(65 + i, 18 * Ls_step, wg_step * i - 4 * wg_step, -1)

    # elements
    element('elasticBeamColumn', 62, 19, 69, areaCB, ECB, Gc30, TCB, IyCB, IzCB, 1)
    for i in range(8):
        element('elasticBeamColumn', 63 + i, 65 + i, 66 + i, areaCB, ECB, Gc30, TCB, IyCB, IzCB, 3)

    # 桥台端部
    # 桥台L
    # nodes
    for i in range(9):
        node(74 + i, 0, wg_step * i - 4 * wg_step, -1)

    # elements
    for i in range(8):
        element('elasticBeamColumn', 71 + i, 74 + i, 75 + i, areaCB, ECB, Gc30, TCB, IyCB, IzCB, 3)

    # 桥台R
    # nodes
    for i in range(9):
        node(83 + i, 18 * Ls_step, wg_step * i - 4 * wg_step, -1)

    # elements
    for i in range(8):
        element('elasticBeamColumn', 79 + i, 83 + i, 84 + i, areaCB, ECB, Gc30, TCB, IyCB, IzCB, 3)

    # 碰撞单元 (一侧两个单元)
    # K_impact_1 = 587000/2
    # K_impact_2 = 202000/2
    # Y_D = 2.5*0.001
    Gap = 23.3 * 0.001

    # 定义碰撞材料
    uniaxialMaterial('ImpactMaterial', 101, K_impact_1, K_impact_2, -dy_impact, -Gap)

    # 桥台L——两个碰撞单元
    element('zeroLength', 87, 56, 74, '-mat', 101, '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)
    element('zeroLength', 88, 64, 82, '-mat', 101, '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)

    # 桥台R——两个碰撞单元
    element('zeroLength', 89, 65, 83, '-mat', 101, '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)
    element('zeroLength', 90, 73, 91, '-mat', 101, '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)

    # 回填土端部地面
    # 回填土L
    for i in range(9):
        node(92 + i, 0, wg_step * i - 4 * wg_step, -1)

    # 回填土R
    for i in range(9):
        node(101 + i, 18 * Ls_step, wg_step * i - 4 * wg_step, -1)

    # 桥台——回填土模型
    # 模型参数
    # 一侧桥台三个单元，除以3
    Kmax_abutment = 21.75 * 1000 * 17.37 / 3
    # 卸载刚度与加载一致
    Kcur_abutment = Kmax_abutment

    # H单位转成inch计算
    H_abutment = 3.59 * 39.37
    # F————>kip/ft * ft * 1.2 ————> *4.48 = kN
    # 除以3
    Fult_abutment = ((((8 * 0.05 * H_abutment) / (1 + 3 * 0.05 * H_abutment)) * (H_abutment ** 1.5)) * 17.37 * 3.28 * 1.2 * 4.448) / 3

    uniaxialMaterial('HyperbolicGapMaterial', 102, Kmax_abutment, Kcur_abutment, Rf_abutment, -Fult_abutment,
                     -Gap_abutment)

    # 桥台L——三个回填土桥台单元
    for i in range(3):
        element('zeroLength', 91 + i, 74 + 4 * i, 92 + 4 * i, '-mat', 102, '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)

    # 桥台R——三个回填土桥台单元
    for i in range(3):
        element('zeroLength', 94 + i, 83 + 4 * i, 101 + 4 * i, '-mat', 102, '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)

    # 桥台——pile
    # 模型参数
    # 桩的数量
    n_pile = 17.37 * 1.2/1.91
    # keff (一侧两个单元）
    keff_abutment_pile = 7 * 1000 * n_pile / 2
    # K1 = 2.333 keff
    k1_abutment_pile = 2.333 * keff_abutment_pile
    # F1
    F1_abutment_pile = k1_abutment_pile * x1a_abutment_pile

    # K2 = 0.428 keff
    k2_abutment_pile = 0.428 * keff_abutment_pile
    # F2
    F2_abutment_pile = F1_abutment_pile + k2_abutment_pile * (x2a_abutment_pile - x1a_abutment_pile)

    uniaxialMaterial('Hysteretic', 103, F1_abutment_pile, x1a_abutment_pile, F2_abutment_pile, x2a_abutment_pile,
                     F2_abutment_pile, 2 * x2a_abutment_pile,
                     -F1_abutment_pile, -x1a_abutment_pile, -F2_abutment_pile, -x2a_abutment_pile, -F2_abutment_pile,
                     -2 * x2a_abutment_pile,
                     pinchx_abutment_pile, pinchy_abutment_pile, 0, 0, 0)

    # 桥台L——两个桩单元
    element('zeroLength', 97, 76, 94, '-mat', 103, 103, '-dir', 1, 2, '-orient', 1, 0, 0, 0, 1, 0)
    element('zeroLength', 98, 80, 98, '-mat', 103, 103, '-dir', 1, 2, '-orient', 1, 0, 0, 0, 1, 0)

    # 桥台R——两个桩单元
    element('zeroLength', 99, 85, 103, '-mat', 103, 103, '-dir', 1, 2, '-orient', 1, 0, 0, 0, 1, 0)
    element('zeroLength', 100, 89, 107, '-mat', 103, 103, '-dir', 1, 2, '-orient', 1, 0, 0, 0, 1, 0)

    '''-------------------pier bearing & retainer--------------------'''

    # start from node 110
    # start from element 101

    # bearing
    # 支座基本材料性能
    # 抗压
    uniaxialMaterial("Elastic", 1, 1.30E+06)
    # 扭矩
    uniaxialMaterial("Elastic", 2, 0)
    # 弯矩
    uniaxialMaterial("Elastic", 3, 8.23E+03)
    # 剪切刚度
    Kb1 = 2.95 * 1000
    # 摩擦系数
    frictionModel('Coulomb', 1, 0.2)

    # 支座——墩1#
    # 支座
    # for i in range(3):
    #     node(1001 + i, 6 * Ls_step, wg_step * (i + 1) - 4 * wg_step, -1)
    #     node(1004 + i, 6 * Ls_step, wg_step * (i + 5) - 4 * wg_step, -1)
    #     element('elasticBeamColumn', )
    #
    # # 支座下部
    # for i in range(3):
    #     node(1007 + i, 6 * Ls_step, wg_step * (i + 1) - 4 * wg_step, -1)
    #     node(1010 + i, 6 * Ls_step, wg_step * (i + 5) - 4 * wg_step, -1)


    for i in range(3):
        element('flatSliderBearing', 101 + i, 39 + i, 21 + i, 1, Kb1, '-P', 1, '-T', 2, '-My', 3, '-Mz', 3, '-orient',
                0, 0, 1, 1, 0, 0)
        element('flatSliderBearing', 104 + i, 43 + i, 25 + i, 1, Kb1, '-P', 1, '-T', 2, '-My', 3, '-Mz', 3, '-orient',
                0, 0, 1, 1, 0, 0)

    for i in range(3):
        equalDOF(39 + i, 21 + i, 4, 5, 6)
        equalDOF(43 + i, 25 + i, 4, 5, 6)

    # 耦合
    # for i in range(3):
    #     equalDOF(1001 + i, 21 + i, 1, 2, 3, 4, 5, 6)
    #     equalDOF(1004 + i, 25 + i, 1, 2, 3, 4, 5, 6)
    #     equalDOF(1007 + i, 39 + i, 1, 2, 3, 4, 5, 6)
    #     equalDOF(1010 + i, 43 + i, 1, 2, 3, 4, 5, 6)

    # 支座——墩2#
    for i in range(3):
        element('flatSliderBearing', 107 + i, 48 + i, 30 + i, 1, Kb1, '-P', 1, '-T', 2, '-My', 3, '-Mz', 3, '-orient',
                0, 0, 1, 1, 0, 0)
        element('flatSliderBearing', 110 + i, 52 + i, 34 + i, 1, Kb1, '-P', 1, '-T', 2, '-My', 3, '-Mz', 3, '-orient',
                0, 0, 1, 1, 0, 0)

    for i in range(3):
        equalDOF(48 + i, 30 + i, 4, 5, 6)
        equalDOF(52 + i, 34 + i, 4, 5, 6)

    # 支座——桥台L
    for i in range(3):
        element('flatSliderBearing', 113 + i, 75 + i, 57 + i, 1, Kb1, '-P', 1, '-T', 2, '-My', 3, '-Mz', 3, '-orient',
                0, 0, 1, 1, 0, 0)
        element('flatSliderBearing', 116 + i, 79 + i, 61 + i, 1, Kb1, '-P', 1, '-T', 2, '-My', 3, '-Mz', 3, '-orient',
                0, 0, 1, 1, 0, 0)

    for i in range(3):
        equalDOF(75 + i, 57 + i, 4, 5, 6)
        equalDOF(79 + i, 61 + i, 4, 5, 6)

    # 支座——桥台R
    for i in range(3):
        element('flatSliderBearing', 119 + i, 84 + i, 66 + i, 1, Kb1, '-P', 1, '-T', 2, '-My', 3, '-Mz', 3, '-orient',
                0, 0, 1, 1, 0, 0)
        element('flatSliderBearing', 122 + i, 88 + i, 70 + i, 1, Kb1, '-P', 1, '-T', 2, '-My', 3, '-Mz', 3, '-orient',
                0, 0, 1, 1, 0, 0)

    for i in range(3):
        equalDOF(84 + i, 66 + i, 4, 5, 6)
        equalDOF(88 + i, 70 + i, 4, 5, 6)

    # retainer
    # 挡块性能参数
    # 屈服力
    Fdy = 49
    # 第一刚度
    Kd1 = 4.07 * 1000
    # 硬化系数
    b_retainer = 0.08

    uniaxialMaterial('Steel02', 4, Fdy, Kd1, b_retainer)
    #
    uniaxialMaterial('Steel02', 5, 2 * Fdy, 2 * Kd1, b_retainer)

    # 墩1
    element('zeroLength', 125, 38, 20, '-mat', 4, '-dir', 2, '-orient', 1, 0, 0, 0, 1, 0)
    element('zeroLength', 126, 46, 28, '-mat', 4, '-dir', 2, '-orient', 1, 0, 0, 0, 1, 0)

    equalDOF(38, 20, 4, 5, 6)
    equalDOF(46, 28, 4, 5, 6)

    # 墩2
    element('zeroLength', 127, 47, 29, '-mat', 4, '-dir', 2, '-orient', 1, 0, 0, 0, 1, 0)
    element('zeroLength', 128, 55, 37, '-mat', 4, '-dir', 2, '-orient', 1, 0, 0, 0, 1, 0)

    equalDOF(47, 29, 4, 5, 6)
    equalDOF(55, 37, 4, 5, 6)

    # 桥台L
    element('zeroLength', 129, 78, 60, '-mat', 5, '-dir', 2, '-orient', 1, 0, 0, 0, 1, 0)
    equalDOF(78, 60, 4, 5, 6)

    # 桥台R
    element('zeroLength', 130, 87, 69, '-mat', 5, '-dir', 2, '-orient', 1, 0, 0, 0, 1, 0)
    equalDOF(87, 69, 4, 5, 6)

    '''--------------------pier-------------------------'''
    # start from node 110
    # start from element 131

    # 墩nodes
    # 依据墩高确定节点划分
    nodes_Pier1 = 8

    # 桥墩节点长度
    Pier_length = 8.8 / nodes_Pier1

    # 墩1
    for i in range(nodes_Pier1):
        node(110 + i, 6 * Ls_step, -2 * wg_step, -1 - (i + 1) * Pier_length)
        node(110 + nodes_Pier1 + i, 6 * Ls_step, 2 * wg_step, -1 - (i + 1) * Pier_length)
    nodes_Pier2_start = 110 + 2 * nodes_Pier1

    # 墩2
    for i in range(nodes_Pier1):
        node(nodes_Pier2_start + i, 12 * Ls_step, -2 * wg_step, -1 - (i + 1) * Pier_length)
        node(nodes_Pier2_start + nodes_Pier1 + i, 12 * Ls_step, 2 * wg_step, -1 - (i + 1) * Pier_length)

    # 桥墩1钢臂
    element('elasticBeamColumn', 131, 40, 110, areaRigid2, ERigid, GRigid, TRigid2, IyRigid2, IzRigid2, 1)
    element('elasticBeamColumn', 132, 44, 110 + nodes_Pier1, areaRigid2, ERigid, GRigid, TRigid2, IyRigid2, IzRigid2, 1)

    # 桥墩2钢臂
    element('elasticBeamColumn', 133, 49, nodes_Pier2_start, areaRigid2, ERigid, GRigid, TRigid2, IyRigid2, IzRigid2, 1)
    element('elasticBeamColumn', 134, 53, nodes_Pier2_start + nodes_Pier1, areaRigid2, ERigid, GRigid, TRigid2,
            IyRigid2, IzRigid2, 1)

    # 桥墩材料
    # concrete
    # unconfined
    fc_unconfined = 34.5 * 1000  # kN/m^2
    epsc_unconfined = 0.002
    fcu_unconfined = 0.2 * fc_unconfined  # kN/m^2
    epscu_unconfined = 0.004
    # epscu_unconfined = 0.004 + 0.9 * Base_Data_pT[III] * (Base_Data_fy[III] / 300)

    # confined
    # 系数k

    k_concrete = 1 + 0.005 * 420 / 34.5

    fc_confined = k_concrete * fc_unconfined
    epsc_confined = k_concrete * epsc_unconfined
    fcu_confined = 0.8 * k_concrete * fc_unconfined

    # mander 1988
    # epscu_confined = epsc_unconfined * (1 + 5 * (fc_confined/fc_unconfined - 1))

    # 0.02
    epscu_confined = 0.02

    # 系数Zm
    # Zm_confined = 0.5 / (
    #             (3 + 0.29 * Base_Data_fc[III]) / (145 * Base_Data_fc[III] - 1000) + 0.75 * Base_Data_pT[III] * (
    #                 1.5 / 80) ** 0.5 - 0.002 * k_concrete)

    # epscu_confined = 0.8 / Zm_confined + 0.002 * k_concrete

    # 定义材料
    '''茂昌'''
    # # unconfined
    # uniaxialMaterial("Concrete01", 21, -20100, -0.00134, -1133, -6.00E-03)
    # # confined
    # uniaxialMaterial("Concrete01", 22, -23940, -2.60E-03, -2867.183979, -1.59E-02)

    # unconfined
    uniaxialMaterial('Concrete01', 21, -fc_unconfined, -epsc_unconfined, -fcu_unconfined, -epscu_unconfined)
    # confined
    uniaxialMaterial('Concrete01', 22, -fc_confined, -epsc_confined, -fcu_confined, -epscu_confined)

    # steel
    # 参数
    Fy_steel = 420 * 1000
    E0_steel = 201 * 1000000
    b_steel = 0.0083

    uniaxialMaterial('Steel02', 23, Fy_steel, E0_steel, b_steel, 15, 0.925, 0.15)

    # Fiber
    DColumn = 2
    Ec30 = 3.0e7
    Gc30 = 0.4 * Ec30
    TColumn = pi * (DColumn ** 4 / 32)
    GJ = Gc30 * TColumn

    n_layer = 32
    R_unconfined = DColumn / 2
    R_confined = R_unconfined * 14 / 15
    S_column = pi * (R_unconfined ** 2)
    # 单根钢筋截面积
    area_steel = S_column * 0.015 / n_layer

    # 定义截面

    section("Fiber", 1, "-GJ", GJ)
    # 定义 PatchCirc "un"
    patch('circ', 21, 10, 10, 0.0, 0.0, R_confined, R_unconfined, 0.0, 360.0)
    # 定义 PatchCirc "con"
    patch('circ', 22, 10, 10, 0.0, 0.0, 0.0, R_confined, 0.0, 360.0)
    # 定义 LayerCircular "steel"
    layer('circ', 23, n_layer, area_steel, 0.0, 0.0, R_confined, 0.0, 360.0)

    # 沿元素长度的积分点数
    np = 5

    # Lobatto integratoin
    beamIntegration('Lobatto', 1, 1, np)

    # 桥墩
    # 桥墩1 elements
    for i in range(nodes_Pier1 - 1):
        element('dispBeamColumn', 135 + i, 110 + i, 111 + i, 1, 1)
        element('dispBeamColumn', 135 + nodes_Pier1 - 1 + i, 110 + nodes_Pier1 + i, 111 + nodes_Pier1 + i, 1, 1)

    for i in range(nodes_Pier1 - 1):
        element('dispBeamColumn', 135 + 2 * nodes_Pier1 - 2 + i, nodes_Pier2_start + i, nodes_Pier2_start + 1 + i, 1, 1)
        element('dispBeamColumn', 135 + 3 * nodes_Pier1 - 3 + i, nodes_Pier2_start + nodes_Pier1 + i,
                nodes_Pier2_start + nodes_Pier1 + i + 1, 1, 1)

    # 基础
    # nodes
    # 墩1
    node(nodes_Pier2_start + 2 * nodes_Pier1, 6 * Ls_step, -2 * wg_step, -1 - nodes_Pier1 * Pier_length)
    node(nodes_Pier2_start + 2 * nodes_Pier1 + 1, 6 * Ls_step, 2 * wg_step, -1 - nodes_Pier1 * Pier_length)

    # 墩2
    node(nodes_Pier2_start + 2 * nodes_Pier1 + 2, 12 * Ls_step, -2 * wg_step, -1 - nodes_Pier1 * Pier_length)
    node(nodes_Pier2_start + 2 * nodes_Pier1 + 3, 12 * Ls_step, 2 * wg_step, -1 - nodes_Pier1 * Pier_length)

    # 材料参数
    # 水平弹簧
    uniaxialMaterial('Elastic', 201, 175.1 * 1000)
    # 旋转弹簧
    uniaxialMaterial('Elastic', 202, 1.36 * 1e6)
    # 垂直大刚度弹簧
    uniaxialMaterial('Elastic', 203, 1e9)

    # elements
    # 墩1基础
    element('zeroLength', 135 + 4 * nodes_Pier1 - 4, 110 + nodes_Pier1 - 1, nodes_Pier2_start + 2 * nodes_Pier1,
            '-mat', 201, 201, 202, 202, '-dir', 1, 2, 4, 5, '-orient', 1, 0, 0, 0, 1, 0)

    element('zeroLength', 135 + 4 * nodes_Pier1 - 3, 110 + 2 * nodes_Pier1 - 1, nodes_Pier2_start + 2 * nodes_Pier1 + 1,
            '-mat', 201, 201, 202, 202, '-dir', 1, 2, 4, 5, '-orient', 1, 0, 0, 0, 1, 0)

    # 墩2基础
    element('zeroLength', 135 + 4 * nodes_Pier1 - 2, nodes_Pier2_start + nodes_Pier1 - 1,
            nodes_Pier2_start + 2 * nodes_Pier1 + 2,
            '-mat', 201, 201, 202, 202, '-dir', 1, 2, 4, 5, '-orient', 1, 0, 0, 0, 1, 0)

    element('zeroLength', 135 + 4 * nodes_Pier1 - 1, nodes_Pier2_start + 2 * nodes_Pier1 - 1,
            nodes_Pier2_start + 2 * nodes_Pier1 + 3,
            '-mat', 201, 201, 202, 202, '-dir', 1, 2, 4, 5, '-orient', 1, 0, 0, 0, 1, 0)

    # 边界条件
    # 墩1基础
    fix(nodes_Pier2_start + 2 * nodes_Pier1, 1, 1, 1, 1, 1, 1)
    fix(nodes_Pier2_start + 2 * nodes_Pier1 + 1, 1, 1, 1, 1, 1, 1)
    # 墩1底
    fix(110 + nodes_Pier1 - 1, 0, 0, 1, 0, 0, 1)
    fix(110 + 2 * nodes_Pier1 - 1, 0, 0, 1, 0, 0, 1)
    # equalDOF(110 + nodes_Pier1 - 1, nodes_Pier2_start + 2 * nodes_Pier1, 3, 6)
    # equalDOF(110 + 2 * nodes_Pier1 - 1, nodes_Pier2_start + 2 * nodes_Pier1 + 1, 3, 6)

    # 墩2基础
    fix(nodes_Pier2_start + 2 * nodes_Pier1 + 2, 1, 1, 1, 1, 1, 1)
    fix(nodes_Pier2_start + 2 * nodes_Pier1 + 3, 1, 1, 1, 1, 1, 1)
    # 墩2底
    fix(nodes_Pier2_start + nodes_Pier1 - 1, 0, 0, 1, 0, 0, 1)
    fix(nodes_Pier2_start + 2 * nodes_Pier1 - 1, 0, 0, 1, 0, 0, 1)
    # equalDOF(nodes_Pier2_start + nodes_Pier1 - 1, nodes_Pier2_start + 2 * nodes_Pier1 + 2, 3, 6)
    # equalDOF(nodes_Pier2_start + 2 * nodes_Pier1 - 1, nodes_Pier2_start + 2 * nodes_Pier1 + 3, 3, 6)

    # 回填土L & 桥台L自由度耦合
    for i in range(9):
        fix(74 + i, 0, 0, 1, 1, 1, 1)
        # equalDOF(74 + i, 92 + i, 3, 4, 5, 6)

    # 回填土R & 桥台R自由度耦合
    for i in range(9):
        fix(83 + i, 0, 0, 1, 1, 1, 1)
        # equalDOF(83 + i, 101 + i, 3, 4, 5, 6)

    # 回填土L
    for i in range(9):
        fix(92 + i, 1, 1, 1, 1, 1, 1)

    # 回填土R
    for i in range(9):
        fix(101 + i, 1, 1, 1, 1, 1, 1)

    # 质量
    mass_node = 1402.5 / 19
    mass_pier = 8.8 * ((pi * DColumn ** 2)/4) * 25/(4*9.81)
    mass_capbeam = 17.37 * ((pi * DColumn ** 2)/4) * 25/(4*9.81)
    # 不考虑下部结构
    # mass_pier_top = mass_node
    # 考虑下部结构
    mass_pier_top = mass_node + mass_pier + mass_capbeam

    for i in range(6):
        mass(i + 1, mass_node, mass_node, mass_node, 0, 0, 0)
        mass(i + 14, mass_node, mass_node, mass_node, 0, 0, 0)

    for i in range(5):
        mass(i + 8, mass_node, mass_node, mass_node, 0, 0, 0)

    # 墩1顶
    mass(7, mass_pier_top, mass_pier_top, mass_pier_top, 0, 0, 0)
    # 墩1顶
    mass(13, mass_pier_top, mass_pier_top, mass_pier_top, 0, 0, 0)

    # 定义重量荷载
    timeSeries('Linear', 1)
    pattern('Plain', 1, 1)

    # 加载
    Load_mass = 9.81 * 1402.5/7
    Load_pier_top = 9.81 * mass_pier_top

    load(1, 0, 0, -Load_mass, 0, 0, 0)
    load(4, 0, 0, -Load_mass, 0, 0, 0)
    load(10, 0, 0, -Load_mass, 0, 0, 0)
    load(16, 0, 0, -Load_mass, 0, 0, 0)
    load(19, 0, 0, -Load_mass, 0, 0, 0)

    load(7, 0, 0, -Load_pier_top, 0, 0, 0)
    load(13, 0, 0, -Load_pier_top, 0, 0, 0)

    # Create the system of equation, a sparse solver with partial pivoting
    system('BandGeneral')
    # Create the constraint handler, the transformation method
    constraints('Transformation')
    # Create the DOF numberer, the reverse Cuthill-McKee algorithm
    numberer('RCM')
    # Create the convergence test, the norm of the residual with a tolerance of
    # 1e-12 and a max number of iterations of 10
    test('NormDispIncr', 1.0e-12, 10)
    # Create the solution algorithm, a Newton-Raphson algorithm
    algorithm('Newton')
    # Create the integration scheme, the LoadControl scheme using steps of 0.1
    integrator('LoadControl', 1)
    # Create the analysis object
    analysis('Static')
    analyze(1)
    loadConst('-time', 0.0)

    # 计算特征值
    numEigen = 10
    eigenValues = eigen(numEigen)
    # print("eigen values at start of transient:", eigenValues)
    LMD1 = eigenValues[0]
    omega1 = LMD1 ** 0.5
    T0 = 2 * pi / omega1
    print('一阶周期为：', T0)
    LMD2 = eigenValues[1]
    omega2 = LMD2 ** 0.5
    T1 = 2 * pi/omega2
    print('二阶周期为：', T1)

    # print(Base_Data_Ls[III])
    # print(Base_Data_Hpier[III])
    # print(Base_Data_wg[III])
    # ops.plot_model(node_labels=0, element_labels=1)
    # ops.plot_mode_shape(1)
    # ops.plot_mode_shape(2)
    # plt.show()

    # 动力计算
    wipeAnalysis()
    dt = 0.01
    npts = 6000

    # 输入地震动
    os.chdir('./NTHA/pre_2b/1')
    GMfile = f"GM_{III + 1:03d}.txt"

    timeSeries('Path', 2, '-dt', dt, '-filePath', GMfile, '-factor', g)
    pattern('UniformExcitation', 2, 2, '-accel', 2)

    # define && apply damping
    damp = 0.05
    aMswitch = 1.0
    betakswitch = 0
    betaKinitswitch = 0
    betaKcommswitch = 1
    nEigeni = 1
    nEigenj = 2
    lambdaN = eigen(nEigenj)
    lambdai = lambdaN[nEigeni - 1]
    lambdaj = lambdaN[nEigenj - 1]
    omegai = lambdai ** 0.5
    omegaj = lambdaj ** 0.5
    # 计算阻尼系数
    aM = aMswitch * damp * (2 * omegai * omegaj) / (omegai + omegaj)
    betaK = betakswitch * 2 * damp / (omegai + omegaj)
    betaKinit = betaKinitswitch * 2 * damp / (omegai + omegaj)
    betaKcomm = betaKcommswitch * 2 * damp / (omegai + omegaj)

    rayleigh(aM, betaK, betaKinit, betaKcomm)

    # create the analysis
    constraints('Transformation')
    numberer('RCM')
    system('BandGeneral')
    test('NormDispIncr', 1.0e-6, 500)
    algorithm('NewtonLineSearch', 0.8)
    integrator('Newmark', 0.5, 0.25)
    analysis('Transient')

    # 输出
    os.chdir('..')
    os.chdir('./result_1')
    out_girder_D = "%s_girder_D.txt" % (III + 1)
    out_pier_D = "%s_pier_D.txt" % (III + 1)
    out_pier_F = "%s_pier_F.txt" % (III + 1)

    # try
    # recorder('Node', '-file', 'try_node7.txt', '-node', 7, '-dof', 2, 'disp')
    # recorder('Node', '-file', 'try_node42.txt', '-node', 42, '-dof', 2, 'disp')
    # recorder('Element', '-file', 'try_bearing_F.txt', '-ele', 101, 'basicForce')
    # recorder('Element', '-file', 'try_bearing_D.txt', '-ele', 101, 'deformation')
    # recorder('Element', '-file', 'try_retainer_F.txt', '-ele', 125, 'basicForce')
    # recorder('Element', '-file', 'try_retainer_D.txt', '-ele', 125, 'deformation')
    # go!
    recorder('Node', '-file', out_girder_D, '-node', 10, '-dof', 2, 'disp')
    recorder('Node', '-file', out_pier_D, '-node', 42, '-dof', 2, 'disp')
    recorder('Element', '-file', out_pier_F, '-ele', 142, 'globalForce')

    # recorder('Element', '-file', '1g_%s_bearing_F.txt'% (III + 1), '-ele', 101, 'basicForce')
    # recorder('Element', '-file', '1g_%s_bearing_D.txt'% (III + 1), '-ele', 101, 'deformation')
    # recorder('Element', '-file', '1g_%s_retainer_F.txt'% (III + 1), '-ele', 125, 'basicForce')
    # recorder('Element', '-file', '1g_%s_retainer_D.txt'% (III + 1), '-ele', 125, 'deformation')

    TimeSteps = npts + int(30/dt)
    ok = analyze(TimeSteps, dt)
    if ok != 0:
        tfinal = TimeSteps * dt
        tCurrent = getTime()
        ok = 0
        while ok == 0 and tCurrent < tfinal:
            ok = analyze(1, dt)
            if ok != 0:
                print('NewtonLineSearch failed...lets try an initial stiffness for this step')
                test('NormDispIncr', 1.0e-6, 1000, 1)
                algorithm('Newton', initial=True)
                ok = analyze(1, dt)
                if ok == 0:
                    print('that worked... back to NewtonLineSearch')
                    test('NormDispIncr', 1.0e-6, 500, 2)
                    algorithm('NewtonLineSearch')
            if ok != 0:
                print('Try Broyden...')
                algorithm('Broyden')
                ok = analyze(1, dt)
                algorithm('NewtonLineSearch')
            if ok != 0:
                print('Try Newton')
                algorithm('Newton')
                ok = analyze(1, dt)
                algorithm('NewtonLineSearch')
            tCurrent = getTime()

    os.chdir('..')
    os.chdir('..')
    os.chdir('..')