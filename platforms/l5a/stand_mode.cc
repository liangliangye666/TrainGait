#include <thread>

#include "fsm.h"
#include "robot_model.h"
#include "standMode_types.h"
#include "ros_publisher.h"

#include <rclcpp/rclcpp.hpp>

#define _USE_MATH_DEFINES
#include <cmath>

std::string urdf_path = "/sim/model/l5a/urdf/l5aurdf20260209.urdf";
l5a::RobotModel robot_model(urdf_path);
l5a::FSM fsm(robot_model);
bool isCtrl = false;
bool jump_flag = false;
int jump_time = 0;

double call_time_ms = 0.0;  // Initialize time_ms to 0.0
double flag = 0;
auto last_time = std::chrono::high_resolution_clock::now();

struct CtlSigs {
  double vel_des;
  double omega_des;
  bool rl_run;  // 模态切换
};  // 控制信号
CtlSigs ctlSigs{0.0, 0.0, false};
void handleSigs_processing(standmode_output_t* standmode_output, const standmode_input_t* standmode_input, CtlSigs* ctlSigs, l5a::FSM* fsm);
float updateWithSmartBrake(float current, float target, float a_up, float a_down, float dt);

void DisplayObservedValue(standmode_output_t* standmode_output, const Eigen::VectorXd& observed_value);

void standMode_initialize(standmode_output_t* standmode_output, standmode_input_t* standmode_input) {
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  robot_model.Initialize();
  fsm.Initialize(robot_model);
  ctlSigs.rl_run = false;
  isCtrl = false;
  jump_flag = false;
  jump_time = 0;
  return;
}

void SetParameters(standmode_output_t* standmode_output, standmode_input_t* standmode_input) {
  // 使能
  standmode_output->joints_cmd.joint_h_pitch_l.enable = 1;
  standmode_output->joints_cmd.joint_h_roll_l.enable = 1;
  standmode_output->joints_cmd.joint_k_pitch_l.enable = 1;
  standmode_output->joints_cmd.joint_w_pitch_l.enable = 1;

  standmode_output->joints_cmd.joint_h_pitch_r.enable = 1;
  standmode_output->joints_cmd.joint_h_roll_r.enable = 1;
  standmode_output->joints_cmd.joint_k_pitch_r.enable = 1;
  standmode_output->joints_cmd.joint_w_pitch_r.enable = 1;

  // 设置模式：操作模式为1是位置模式，3是速度模式，4是力矩模式，11是力位混合模式
  standmode_output->joints_cmd.joint_h_pitch_l.operation_mode = 4;
  standmode_output->joints_cmd.joint_h_roll_l.operation_mode = 4;
  standmode_output->joints_cmd.joint_k_pitch_l.operation_mode = 4;
  standmode_output->joints_cmd.joint_w_pitch_l.operation_mode = 4;

  standmode_output->joints_cmd.joint_h_pitch_r.operation_mode = 4;
  standmode_output->joints_cmd.joint_h_roll_r.operation_mode = 4;
  standmode_output->joints_cmd.joint_k_pitch_r.operation_mode = 4;
  standmode_output->joints_cmd.joint_w_pitch_r.operation_mode = 4;
}

void standMode_step(standmode_output_t* standmode_output, standmode_input_t* standmode_input) {
  auto call_time = std::chrono::high_resolution_clock::now();
  auto call_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(call_time - last_time);
  double call_time_ns = call_duration.count();
  call_time_ms = call_time_ns / 1000000;

  last_time = call_time;

  SetParameters(standmode_output, standmode_input);

  robot_model.UpdateRealJointStates(standmode_output, standmode_input);
  robot_model.UpdateModel();
  handleSigs_processing(standmode_output, standmode_input, &ctlSigs, &fsm);  // 手柄信号处理
  robot_model.vel_des_ = updateWithSmartBrake(robot_model.vel_des_, ctlSigs.vel_des, 0.5, 1.5, robot_model.control_dt);
  // robot_model.vel_des_ = ctlSigs.vel_des;
  robot_model.omega_des_ = ctlSigs.omega_des;
  Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd pos_cmd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd vel_cmd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd tau_joint = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd pos_fb_kp = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd pos_fb_kd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);

  Eigen::VectorXd tau_gravity = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd tau_fsm = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd pos_fsm = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd kp = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd kd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd qDes = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  Eigen::VectorXd qdotDes = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6);
  // tau_gravity << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  // qDes << 0.3491, 1.8799, 0.0, 0.3491, 1.8799, 0.0;
  // qdotDes << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  // kp << 100.0, 100.0, 0.0, 100.0, 100.0, 0.0;
  // kd << 1.0, 0.8, 1.0, 1.0, 0.8, 1.0;

  // if (ctlSigs.rl_run && !isCtrl) {
  //   isCtrl = true;
  //   std::cout << "Start walk" << std::endl;
  // }

  // for (int i = 0; i < robot_model.pino_model().nv-6; i++) {
  //   tau_joint(i) = tau_gravity(i) + kp(i) * (qDes(i) - robot_model.q_rpy(6 + i)) + kd(i) * (qdotDes(i) - robot_model.qdot(6 + i));
  // }
  flag = 0;

  //jump_cmd
  // if (jump_flag && jump_time<=robot_model.jump_time_window){
  //   robot_model.jump = 1.0;
  //   jump_time++;
  // }else{
  //   robot_model.jump = 0.0;
  //   jump_flag = false;
  //   jump_time = 0;
  // }


    fsm.Run(robot_model);
    tau_fsm = fsm.tau();
    pos_fsm = fsm.pos();
    tau_cmd = tau_fsm;
    pos_cmd = pos_fsm;
    pos_fb_kp = fsm.pos_fb_kp_;
    pos_fb_kd = fsm.pos_fb_kd_;
    flag = 1.0;

  standmode_output->joints_cmd.joint_h_roll_l.KP = pos_fb_kp[static_cast<int>(l5a::Joints::left_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_l.KP = pos_fb_kp[static_cast<int>(l5a::Joints::left_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_l.KP = pos_fb_kp[static_cast<int>(l5a::Joints::left_knee_joint)];
  standmode_output->joints_cmd.joint_w_pitch_l.KP = 0;
  standmode_output->joints_cmd.joint_h_roll_r.KP = pos_fb_kp[static_cast<int>(l5a::Joints::right_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_r.KP = pos_fb_kp[static_cast<int>(l5a::Joints::right_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_r.KP = pos_fb_kp[static_cast<int>(l5a::Joints::right_knee_joint)];
  standmode_output->joints_cmd.joint_w_pitch_r.KP = 0;

  standmode_output->joints_cmd.joint_h_roll_l.KD = pos_fb_kd[static_cast<int>(l5a::Joints::left_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_l.KD = pos_fb_kd[static_cast<int>(l5a::Joints::left_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_l.KD = pos_fb_kd[static_cast<int>(l5a::Joints::left_knee_joint)];
  standmode_output->joints_cmd.joint_w_pitch_l.KD = 0;
  standmode_output->joints_cmd.joint_h_roll_r.KD = pos_fb_kd[static_cast<int>(l5a::Joints::right_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_r.KD = pos_fb_kd[static_cast<int>(l5a::Joints::right_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_r.KD = pos_fb_kd[static_cast<int>(l5a::Joints::right_knee_joint)];
  standmode_output->joints_cmd.joint_w_pitch_r.KD = 0;

  standmode_output->joints_cmd.joint_h_roll_l.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::left_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_l.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::left_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_l.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::left_knee_joint)];
  // standmode_output->joints_cmd.joint_w_pitch_l.vel_cmd = 0;
  standmode_output->joints_cmd.joint_h_roll_r.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::right_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_r.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::right_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_r.pos_cmd = pos_cmd[static_cast<int>(l5a::Joints::right_knee_joint)];
  // standmode_output->joints_cmd.joint_w_pitch_r.vel_cmd = 0;

  // torque
  standmode_output->joints_cmd.joint_h_roll_l.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::left_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_l.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::left_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_l.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::left_knee_joint)];
  standmode_output->joints_cmd.joint_w_pitch_l.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::left_wheel_joint)];

  standmode_output->joints_cmd.joint_h_roll_r.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::right_hip_roll_joint)];
  standmode_output->joints_cmd.joint_h_pitch_r.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::right_hip_pitch_joint)];
  standmode_output->joints_cmd.joint_k_pitch_r.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::right_knee_joint)];
  standmode_output->joints_cmd.joint_w_pitch_r.torque_cmd = tau_cmd[static_cast<int>(l5a::Joints::right_wheel_joint)];

  robot_model.observed_value[1] = robot_model.q_rpy[3];//roll
  robot_model.observed_value[2] = robot_model.q_rpy[4];//pitch
  robot_model.observed_value[3] = robot_model.qdot[3];//roll
  robot_model.observed_value[4] = robot_model.qdot[4];//pitch
  robot_model.observed_value[5] = robot_model.qdot[5];//yaw
  robot_model.observed_value[6] = robot_model.q_rpy[6];
  robot_model.observed_value[7] = robot_model.q_rpy[7];
  robot_model.observed_value[8] = robot_model.q_rpy[8];
  robot_model.observed_value[9] = robot_model.q_rpy[9];
  robot_model.observed_value[10] = robot_model.q_rpy[10];
  robot_model.observed_value[11] = robot_model.q_rpy[11];
  robot_model.observed_value[12] = robot_model.q_rpy[12];
  robot_model.observed_value[13] = robot_model.q_rpy[13];
  // record data
  DisplayObservedValue(standmode_output, robot_model.observed_value);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto step_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - call_time);
  double step_time_ns = step_duration.count();
  double step_time_ms = step_time_ns / 1000000;

  // ros_publisher->Publish("l5a/physics_robot/time/stand_mode_time_ms", call_time_ms);
  // ros_publisher->Publish("l5a/physics_robot/time/step_time_ms", step_time_ms);
}

void standMode_terminate(void) { return; }

void handleSigs_processing(standmode_output_t* standmode_output, const standmode_input_t* standmode_input, CtlSigs* ctlSigs, l5a::FSM* fsm) {
  // 对手柄信号进行处理
  double handle_A = 0;   // 手柄A键
  double handle_RT = 0;  // 手柄RT键
  double handle_B = 0;   // 手柄B键
  double handle_Y = 0;   // 手柄Y键
  double handle_RB = 0;  // 手柄RB键
  double handle_LB = 0;  // 手柄LB键
  double handle_LT = 0;  // 手柄LT键
  if (standmode_input->Handle_signals.a2 >= 0.5) {
    handle_LT = 1.0;
  } else {
    handle_LT = 0;
  }
  if (standmode_input->Handle_signals.a5 >= 0.5) {
    handle_RT = 1.0;
  } else {
    handle_RT = 0;
  }
  if (standmode_input->Handle_signals.b0 >= 0.5) {
    handle_A = 1.0;
  } else {
    handle_A = 0;
  }
  if (standmode_input->Handle_signals.b1 >= 0.5) {
    handle_B = 1.0;
  } else {
    handle_B = 0;
  }
  if (standmode_input->Handle_signals.b3 >= 0.5) {
    handle_Y = 1.0;
  } else {
    handle_Y = 0;
  }
  if (standmode_input->Handle_signals.b4 >= 0.5) {
    handle_LB = 1.0;
  } else {
    handle_LB = 0;
  }
  if (standmode_input->Handle_signals.b5 >= 0.5) {
    handle_RB = 1.0;
  } else {
    handle_RB = 0;
  }

  if (std::abs(standmode_input->Handle_signals.a1 - 0.5) <= 0.1 || handle_LB) {
    ctlSigs->vel_des = 0.0;
  } else {
    ctlSigs->vel_des = (standmode_input->Handle_signals.a1 - 0.5) * (-1.0) * handle_LT;
  }

  if (std::abs(standmode_input->Handle_signals.a3 - 0.5) <= 0.1 || handle_RB) {
    ctlSigs->omega_des = 0.0;
  } else {
    ctlSigs->omega_des = (standmode_input->Handle_signals.a3 - 0.5) * (-1.0) * handle_LT;
  }

  // 定义模态切换按钮
  if (handle_A && handle_B) {
    ctlSigs->rl_run = true;
  }
  if (!handle_Y && handle_A) {
    robot_model.delta_height_des_ = -0.0001;
  }
  if (handle_Y && !handle_A) {
    robot_model.delta_height_des_ = 0.0001;
  }
  if (handle_RT && handle_B) {
    jump_flag = true;
  }


}

void DisplayObservedValue(standmode_output_t* standmode_output, const Eigen::VectorXd& observed_value) {
  standmode_output->observed_value.channel_1 = observed_value[1];
  standmode_output->observed_value.channel_2 = observed_value[2];
  standmode_output->observed_value.channel_3 = observed_value[3];
  standmode_output->observed_value.channel_4 = observed_value[4];
  standmode_output->observed_value.channel_5 = observed_value[5];
  standmode_output->observed_value.channel_6 = observed_value[6];
  standmode_output->observed_value.channel_7 = observed_value[7];
  standmode_output->observed_value.channel_8 = observed_value[8];
  standmode_output->observed_value.channel_9 = observed_value[9];
  standmode_output->observed_value.channel_10 = observed_value[10];
  standmode_output->observed_value.channel_11 = observed_value[11];
  standmode_output->observed_value.channel_12 = observed_value[12];
  standmode_output->observed_value.channel_13 = observed_value[13];
  standmode_output->observed_value.channel_14 = observed_value[14];
  standmode_output->observed_value.channel_15 = observed_value[15];
  standmode_output->observed_value.channel_16 = observed_value[16];
  standmode_output->observed_value.channel_17 = observed_value[17];
  standmode_output->observed_value.channel_18 = observed_value[18];
  standmode_output->observed_value.channel_19 = observed_value[19];
  standmode_output->observed_value.channel_20 = observed_value[20];

  standmode_output->observed_value.channel_21 = observed_value[21];
  standmode_output->observed_value.channel_22 = observed_value[22];
  standmode_output->observed_value.channel_23 = observed_value[23];
  standmode_output->observed_value.channel_24 = observed_value[24];
  standmode_output->observed_value.channel_25 = observed_value[25];
  standmode_output->observed_value.channel_26 = observed_value[26];
  standmode_output->observed_value.channel_27 = observed_value[27];
  standmode_output->observed_value.channel_28 = observed_value[28];
  standmode_output->observed_value.channel_29 = observed_value[29];
  standmode_output->observed_value.channel_30 = observed_value[30];
  standmode_output->observed_value.channel_31 = observed_value[31];
  standmode_output->observed_value.channel_32 = observed_value[32];
  standmode_output->observed_value.channel_33 = observed_value[33];
  standmode_output->observed_value.channel_34 = observed_value[34];
  standmode_output->observed_value.channel_35 = observed_value[35];
  standmode_output->observed_value.channel_36 = observed_value[36];
  standmode_output->observed_value.channel_37 = observed_value[37];
  standmode_output->observed_value.channel_38 = observed_value[38];
  standmode_output->observed_value.channel_39 = observed_value[39];
  standmode_output->observed_value.channel_40 = observed_value[40];

  standmode_output->observed_value.channel_41 = observed_value[41];
  standmode_output->observed_value.channel_42 = observed_value[42];
  standmode_output->observed_value.channel_43 = observed_value[43];
  standmode_output->observed_value.channel_44 = observed_value[44];
  standmode_output->observed_value.channel_45 = observed_value[45];
  standmode_output->observed_value.channel_46 = observed_value[46];
  standmode_output->observed_value.channel_47 = observed_value[47];
  standmode_output->observed_value.channel_48 = observed_value[48];
  standmode_output->observed_value.channel_49 = observed_value[49];
  standmode_output->observed_value.channel_50 = observed_value[50];
  standmode_output->observed_value.channel_51 = observed_value[51];
  standmode_output->observed_value.channel_52 = observed_value[52];
  standmode_output->observed_value.channel_53 = observed_value[53];
  standmode_output->observed_value.channel_54 = observed_value[54];
  standmode_output->observed_value.channel_55 = observed_value[55];
  standmode_output->observed_value.channel_56 = observed_value[56];
  standmode_output->observed_value.channel_57 = observed_value[57];
  standmode_output->observed_value.channel_58 = observed_value[58];
  standmode_output->observed_value.channel_59 = observed_value[59];
  standmode_output->observed_value.channel_60 = observed_value[60];

  standmode_output->observed_value.channel_61 = observed_value[61];
  standmode_output->observed_value.channel_62 = observed_value[62];
  standmode_output->observed_value.channel_63 = observed_value[63];
  standmode_output->observed_value.channel_64 = observed_value[64];
  standmode_output->observed_value.channel_65 = observed_value[65];
  standmode_output->observed_value.channel_66 = observed_value[66];
  standmode_output->observed_value.channel_67 = observed_value[67];
  standmode_output->observed_value.channel_68 = observed_value[68];
  standmode_output->observed_value.channel_69 = observed_value[69];
  standmode_output->observed_value.channel_70 = observed_value[70];
  standmode_output->observed_value.channel_71 = observed_value[71];
  standmode_output->observed_value.channel_72 = observed_value[72];
  standmode_output->observed_value.channel_73 = observed_value[73];
  standmode_output->observed_value.channel_74 = observed_value[74];
  standmode_output->observed_value.channel_75 = observed_value[75];
  standmode_output->observed_value.channel_76 = observed_value[76];
  standmode_output->observed_value.channel_77 = observed_value[77];
  // standmode_output->observed_value.channel_78 = observed_value[78];//78通道控制支撑架
  standmode_output->observed_value.channel_79 = observed_value[79];
  standmode_output->observed_value.channel_80 = observed_value[80];
}

float updateWithSmartBrake(float current, float target, float a_up, float a_down, float dt) {
  float dv = target - current;
  float step;

  // 情况 1：符号不同（异号），说明要先刹到 0
  if (current * target < 0) {
    // 强制把目标临时设为 0，用大减速度逼近
    float dv0 = 0 - current;
    step = a_down * dt;

    if (fabs(dv0) <= step) {
      return 0.0f;  // 到零
    } else {
      return current + step * (dv0 > 0 ? 1 : -1);
    }
  }

  // 情况 2：符号相同
  if (fabs(target) > fabs(current)) {
    // 加速
    step = a_up * dt;
  } else if (fabs(target) < fabs(current)) {
    // 减速
    step = a_down * dt;
  } else {
    return target;
  }

  if (fabs(dv) <= step) {
    return target;
  } else {
    return current + step * (dv > 0 ? 1 : -1);
  }
}