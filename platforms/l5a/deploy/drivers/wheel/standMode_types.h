/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * File: standMode_types.h
 *
 * Code generated for Simulink model 'standMode'.
 *
 * Model version                  : 5.49
 * Simulink Coder version         : 24.2 (R2024b) 21-Jun-2024
 * C/C++ source code generated on : Tue Feb 25 18:33:37 2025
 *
 * Target selection: ert.tlc
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objective: Execution efficiency
 * Validation result: Not run
 */

#ifndef standMode_types_h_
#define standMode_types_h_
#include "rtwtypes.h"
#ifndef DEFINED_TYPEDEF_FOR_Handle_signals_t_
#define DEFINED_TYPEDEF_FOR_Handle_signals_t_

typedef struct {
  real_T a0;
  real_T a1;
  real_T a2;
  real_T a3;
  real_T a4;
  real_T a5;
  real_T a6;
  real_T a7;
  real_T b0;
  real_T b1;
  real_T b2;
  real_T b3;
  real_T b4;
  real_T b5;
  real_T b6;
  real_T b7;
  real_T b8;
} Handle_signals_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_IMU_signals_t_
#define DEFINED_TYPEDEF_FOR_IMU_signals_t_

typedef struct {
  real_T IMU_wx_body;
  real_T IMU_wy_body;
  real_T IMU_wz_body;
  real_T IMU_accx;
  real_T IMU_accy;
  real_T IMU_accz;
  real_T IMU_wx_world;
  real_T IMU_wy_world;
  real_T IMU_wz_world;
  real_T IMU_roll;
  real_T IMU_pitch;
  real_T IMU_yaw;
  real_T IMU_qw;
  real_T IMU_qx;
  real_T IMU_qy;
  real_T IMU_qz;
} IMU_signals_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_joint_status_t_
#define DEFINED_TYPEDEF_FOR_joint_status_t_

typedef struct {
  real_T pos_fb;
  real_T vel_fb;
  real_T torque_fb;
  real_T current_fb;
  real_T motor_temper;
  real_T min_soft_position_limit;
  real_T max_soft_position_limit;
  real_T max_profile_velocity;
  int8_T enabled;
  int8_T error;
  int8_T operation_mode_display;
  real_T voltage;
} joint_status_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_joints_status_t_
#define DEFINED_TYPEDEF_FOR_joints_status_t_

typedef struct {
  joint_status_t joint_h_pitch_l;
  joint_status_t joint_h_roll_l;
  joint_status_t joint_h_yaw_l;
  joint_status_t joint_k_pitch_l;
  joint_status_t joint_w_pitch_l;
  joint_status_t joint_h_pitch_r;
  joint_status_t joint_h_roll_r;
  joint_status_t joint_h_yaw_r;
  joint_status_t joint_k_pitch_r;
  joint_status_t joint_w_pitch_r;
  joint_status_t joint_arm_l1;
  joint_status_t joint_arm_l2;
  joint_status_t joint_arm_l3;
  joint_status_t joint_arm_l4;
  joint_status_t joint_arm_l5;
  joint_status_t joint_arm_l6;
  joint_status_t joint_arm_l7;
  joint_status_t joint_arm_r1;
  joint_status_t joint_arm_r2;
  joint_status_t joint_arm_r3;
  joint_status_t joint_arm_r4;
  joint_status_t joint_arm_r5;
  joint_status_t joint_arm_r6;
  joint_status_t joint_arm_r7;
} joints_status_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_arms_status_t_
#define DEFINED_TYPEDEF_FOR_arms_status_t_

typedef struct {
  real_T mass_sum;
  real_T mass_center_sum[3];
} arms_status_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_PID_Para_t_
#define DEFINED_TYPEDEF_FOR_PID_Para_t_

typedef struct {
  real_T IMU_MNoiseCov;
  real_T V_des_max;
  real_T W_des_max;
  real_T initChairX;
  real_T initChairZ;
  real_T MNoiseCov;
  real_T IMU_pitch_offset;
  real_T PID_balance_P;
  real_T PID_balance_I;
  real_T PID_balance_D;
  real_T PID_drive_P;
  real_T PID_drive_I;
  real_T PID_drive_D;
  real_T PID_steer_P;
  real_T PID_steer_I;
  real_T PID_steer_D;
  real_T PID_roll_P;
  real_T PID_roll_I;
  real_T PID_roll_D;
  real_T PID_pitch_P;
  real_T PID_pitch_I;
  real_T PID_pitch_D;
  real_T PID_chairX_P;
  real_T PID_chairX_I;
  real_T PID_chairX_D;
  real_T PID_chairZ_P;
  real_T PID_chairZ_I;
  real_T PID_chairZ_D;
  real_T Para_1;
  real_T Para_2;
  real_T Para_3;
  real_T Para_4;
  real_T Para_5;
  real_T Para_6;
  real_T Para_7;
  real_T Para_8;
  real_T Para_9;
  real_T Para_10;
  real_T Para_11;
  real_T Para_12;
  real_T Para_13;
  real_T Para_14;
  real_T Para_15;
  real_T Para_16;
  real_T Para_17;
  real_T Para_18;
  real_T Para_19;
  real_T Para_20;
  real_T Para_21;
  real_T Para_22;
  real_T Para_23;
  real_T Para_24;
  real_T Para_25;
  real_T Para_26;
  real_T Para_27;
  real_T Para_28;
  real_T Para_29;
  real_T Para_30;
  real_T Para_31;
  real_T Para_32;
  real_T Para_33;
  real_T Para_34;
  real_T Para_35;
  real_T Para_36;
  real_T Para_37;
  real_T Para_38;
  real_T Para_39;
  real_T Para_40;
  real_T Para_41;
  real_T Para_42;
  real_T Para_43;
  real_T Para_44;
  real_T Para_45;
  real_T Para_46;
  real_T Para_47;
  real_T Para_48;
  real_T Para_49;
  real_T Para_50;
  real_T Para_51;
  real_T Para_52;
  real_T Para_53;
  real_T Para_54;
  real_T Para_55;
  real_T Para_56;
  real_T Para_57;
  real_T Para_58;
  real_T Para_59;
  real_T Para_60;
  real_T Para_61;
  real_T Para_62;
  real_T Para_63;
  real_T Para_64;
  real_T Para_65;
  real_T Para_66;
  real_T Para_67;
  real_T Para_68;
  real_T Para_69;
  real_T Para_70;
  real_T Para_71;
  real_T Para_72;
  real_T Para_73;
  real_T Para_74;
  real_T Para_75;
  real_T Para_76;
  real_T Para_77;
  real_T Para_78;
  real_T Para_79;
  real_T Para_80;
  real_T Para_81;
  real_T Para_82;
  real_T Para_83;
  real_T Para_84;
  real_T Para_85;
  real_T Para_86;
  real_T Para_87;
  real_T Para_88;
  real_T Para_89;
  real_T Para_90;
  real_T Para_91;
  real_T Para_92;
  real_T Para_93;
  real_T Para_94;
  real_T Para_95;
  real_T Para_96;
  real_T Para_97;
  real_T Para_98;
  real_T Para_99;
  real_T Para_100;
} PID_Para_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_mass_sensors_t_
#define DEFINED_TYPEDEF_FOR_mass_sensors_t_

typedef struct {
  int32_T error;
  int32_T mass_01;
  int32_T mass_02;
  int32_T mass_03;
  int32_T mass_04;
  int32_T mass_05;
  int32_T mass_06;
  int32_T mass_07;
  int32_T mass_08;
} mass_sensors_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_Navi_signals_input_t_
#define DEFINED_TYPEDEF_FOR_Navi_signals_input_t_

typedef struct {
  real_T vDes;
  real_T wDes;
} Navi_signals_input_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_standmode_input_t_
#define DEFINED_TYPEDEF_FOR_standmode_input_t_

typedef struct {
  Handle_signals_t Handle_signals;
  IMU_signals_t IMU_signals;
  joints_status_t joints_status;
  arms_status_t arms_status;
  PID_Para_t PID_Para;
  mass_sensors_t mass_signals;
  Navi_signals_input_t Navi_signals_input;
  real_T runner_vel_fb;
  real_T Batt_capacity;
} standmode_input_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_IMU_signals_out_
#define DEFINED_TYPEDEF_FOR_IMU_signals_out_

typedef struct {
  real_T IMU_pitch;
  real_T IMU_roll;
  real_T IMU_wy;
  real_T IMU_wz;
} IMU_signals_out;

#endif

#ifndef DEFINED_TYPEDEF_FOR_observed_value_t_
#define DEFINED_TYPEDEF_FOR_observed_value_t_

typedef struct {
  real_T channel_1;
  real_T channel_2;
  real_T channel_3;
  real_T channel_4;
  real_T channel_5;
  real_T channel_6;
  real_T channel_7;
  real_T channel_8;
  real_T channel_9;
  real_T channel_10;
  real_T channel_11;
  real_T channel_12;
  real_T channel_13;
  real_T channel_14;
  real_T channel_15;
  real_T channel_16;
  real_T channel_17;
  real_T channel_18;
  real_T channel_19;
  real_T channel_20;
  real_T channel_21;
  real_T channel_22;
  real_T channel_23;
  real_T channel_24;
  real_T channel_25;
  real_T channel_26;
  real_T channel_27;
  real_T channel_28;
  real_T channel_29;
  real_T channel_30;
  real_T channel_31;
  real_T channel_32;
  real_T channel_33;
  real_T channel_34;
  real_T channel_35;
  real_T channel_36;
  real_T channel_37;
  real_T channel_38;
  real_T channel_39;
  real_T channel_40;
  real_T channel_41;
  real_T channel_42;
  real_T channel_43;
  real_T channel_44;
  real_T channel_45;
  real_T channel_46;
  real_T channel_47;
  real_T channel_48;
  real_T channel_49;
  real_T channel_50;
  real_T channel_51;
  real_T channel_52;
  real_T channel_53;
  real_T channel_54;
  real_T channel_55;
  real_T channel_56;
  real_T channel_57;
  real_T channel_58;
  real_T channel_59;
  real_T channel_60;
  real_T channel_61;
  real_T channel_62;
  real_T channel_63;
  real_T channel_64;
  real_T channel_65;
  real_T channel_66;
  real_T channel_67;
  real_T channel_68;
  real_T channel_69;
  real_T channel_70;
  real_T channel_71;
  real_T channel_72;
  real_T channel_73;
  real_T channel_74;
  real_T channel_75;
  real_T channel_76;
  real_T channel_77;
  real_T channel_78;
  real_T channel_79;
  real_T channel_80;
} observed_value_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_Navi_signals_output_t_
#define DEFINED_TYPEDEF_FOR_Navi_signals_output_t_

typedef struct {
  real_T vAct;
  real_T wAct;
} Navi_signals_output_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_joint_cmd_t_
#define DEFINED_TYPEDEF_FOR_joint_cmd_t_

typedef struct {
  real_T pos_cmd;
  real_T vel_cmd;
  real_T torque_cmd;
  real_T current_cmd;
  real_T velocity;
  real_T max_velocity;
  real_T acceleration;
  real_T deceleration;
  real_T KP;
  real_T KD;
  int8_T enable;
  int8_T operation_mode;
  uint8_T polarity;
} joint_cmd_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_joints_cmd_t_
#define DEFINED_TYPEDEF_FOR_joints_cmd_t_

typedef struct {
  joint_cmd_t joint_h_pitch_l;
  joint_cmd_t joint_h_roll_l;
  joint_cmd_t joint_h_yaw_l;
  joint_cmd_t joint_k_pitch_l;
  joint_cmd_t joint_w_pitch_l;
  joint_cmd_t joint_h_pitch_r;
  joint_cmd_t joint_h_roll_r;
  joint_cmd_t joint_h_yaw_r;
  joint_cmd_t joint_k_pitch_r;
  joint_cmd_t joint_w_pitch_r;
  joint_cmd_t ArmL_J1_Cmd;
  joint_cmd_t ArmL_J2_Cmd;
  joint_cmd_t ArmL_J3_Cmd;
  joint_cmd_t ArmL_J4_Cmd;
  joint_cmd_t ArmL_J5_Cmd;
  joint_cmd_t ArmL_J6_Cmd;
  joint_cmd_t ArmL_J7_Cmd;
  joint_cmd_t ArmR_J1_Cmd;
  joint_cmd_t ArmR_J2_Cmd;
  joint_cmd_t ArmR_J3_Cmd;
  joint_cmd_t ArmR_J4_Cmd;
  joint_cmd_t ArmR_J5_Cmd;
  joint_cmd_t ArmR_J6_Cmd;
  joint_cmd_t ArmR_J7_Cmd;
} joints_cmd_t;

#endif

#ifndef DEFINED_TYPEDEF_FOR_standmode_output_t_
#define DEFINED_TYPEDEF_FOR_standmode_output_t_

typedef struct {
  observed_value_t observed_value;
  joints_cmd_t joints_cmd;
  Navi_signals_output_t Navi_signals_output;
} standmode_output_t;

#endif

/* Custom Type definition for MATLAB Function: '<S1>/MATLAB Function8' */
#ifndef struct_tag_stRJlYAl3pjTyFPnByxQ8b
#define struct_tag_stRJlYAl3pjTyFPnByxQ8b

struct tag_stRJlYAl3pjTyFPnByxQ8b
{
  real_T Process_StepInput;
  real_T EstimationData;
};

#endif                                 /* struct_tag_stRJlYAl3pjTyFPnByxQ8b */

#ifndef typedef_stRJlYAl3pjTyFPnByxQ8b_standM_T
#define typedef_stRJlYAl3pjTyFPnByxQ8b_standM_T

typedef struct tag_stRJlYAl3pjTyFPnByxQ8b stRJlYAl3pjTyFPnByxQ8b_standM_T;

#endif                             /* typedef_stRJlYAl3pjTyFPnByxQ8b_standM_T */

/* Custom Type definition for MATLAB Function: '<S1>/MATLAB Function20' */
#ifndef struct_tag_s9jYAhnLwYvBxffH4z6wNKG
#define struct_tag_s9jYAhnLwYvBxffH4z6wNKG

struct tag_s9jYAhnLwYvBxffH4z6wNKG
{
  real_T x_est[3];
  real_T Eye[9];
  real_T P_est[9];
  real_T Q[9];
  real_T R[9];
  real_T A[9];
  real_T H[9];
};

#endif                                 /* struct_tag_s9jYAhnLwYvBxffH4z6wNKG */

#ifndef typedef_s9jYAhnLwYvBxffH4z6wNKG_stand_T
#define typedef_s9jYAhnLwYvBxffH4z6wNKG_stand_T

typedef struct tag_s9jYAhnLwYvBxffH4z6wNKG s9jYAhnLwYvBxffH4z6wNKG_stand_T;

#endif                             /* typedef_s9jYAhnLwYvBxffH4z6wNKG_stand_T */

/* Custom Type definition for MATLAB Function: '<S1>/MATLAB Function13' */
#ifndef struct_tag_sws399s4figUQr0kdVXer9B
#define struct_tag_sws399s4figUQr0kdVXer9B

struct tag_sws399s4figUQr0kdVXer9B
{
  real_T x_est[6];
  real_T Eye[36];
  real_T P_est[36];
  real_T Q[36];
  real_T R[36];
  real_T A[36];
  real_T H[36];
};

#endif                                 /* struct_tag_sws399s4figUQr0kdVXer9B */

#ifndef typedef_sws399s4figUQr0kdVXer9B_stand_T
#define typedef_sws399s4figUQr0kdVXer9B_stand_T

typedef struct tag_sws399s4figUQr0kdVXer9B sws399s4figUQr0kdVXer9B_stand_T;

#endif                             /* typedef_sws399s4figUQr0kdVXer9B_stand_T */

/* Forward declaration for rtModel */
typedef struct tag_RTM_standMode_T RT_MODEL_standMode_T;

#endif                                 /* standMode_types_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
