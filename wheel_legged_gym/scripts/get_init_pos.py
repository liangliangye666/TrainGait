import numpy as np
import pinocchio as pin
from pathlib import Path
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR

def main():
    """
    使用 Pinocchio 计算机器人平衡姿态的主函数
    """
    # ==================== 1. 设置模型路径 ====================
    urdf_path = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/l5a/urdf/l5aurdf20260209.urdf"
    # ==================== 2. 从 URDF 加载模型 ====================
    try:
        # 从 URDF 文件构建模型
        joint_model = pin.JointModelFreeFlyer()
        model = pin.buildModelFromUrdf(urdf_path, joint_model)
        print(f"✅ 成功加载模型: {model.name}")
        print(f"   自由度 (nv): {model.nv}")
        print(f"   关节数量: {model.njoints}")
        for i, name in enumerate(model.names):
            print(i, name)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # ==================== 3. 创建数据对象 ====================
    data = model.createData()
    q = np.zeros(model.nq)
    
    # ==================== 4. 计算质心偏角 ====================
    pos_left_wheel_id = model.getJointId("left_wheel_joint")
    pos_right_wheel_id = model.getJointId("right_wheel_joint")
    print("id l r", pos_left_wheel_id, pos_right_wheel_id)
    angHip = 40.0 / 180.0 * np.pi
    for angKnee in np.arange(-120.0/180.0*np.pi, 120.0/180.0*np.pi, 0.01 /180.0*np.pi):
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, angHip, angKnee, 0.0, 0.0, angHip, angKnee, 0.0], dtype=np.float64)
        pin.forwardKinematics(model, data, q)
        pos_left_wheel = data.oMi[pos_left_wheel_id].translation # 左轮的位置
        pos_right_wheel = data.oMi[pos_right_wheel_id].translation # 右轮的位置
        # print("pos_wheel", pos_left_wheel)
        pin.centerOfMass(model, data, q)
        com = data.com[0]
        # print("pos_com", com)
        vec_wheel2com = com - pos_left_wheel
        # print("vec", vec_wheel2com)
        comAng = np.arctan2(vec_wheel2com[0], vec_wheel2com[2])
        # print("comAng", comAng)
        if np.abs(comAng) <= 0.01 / 180.0 * np.pi:
            print("comAng", comAng)
            print("angHip(rad/degree)", angHip, angHip / np.pi * 180.0)
            print("angKnee(rad/degree)", angKnee, angKnee / np.pi * 180.0)
            print("init_pos_l", pos_left_wheel)
            print("init_pos_r", pos_right_wheel)
            print("height", pos_right_wheel[2]-0.127)
            break

if __name__ == "__main__":
    main()