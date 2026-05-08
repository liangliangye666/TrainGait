#!/bin/bash

work_path=`pwd`

#source /opt/ros/foxy/setup.bash
#/bin/bash -c "source $work_path/scripts/set_env.sh && $work_path/double_wheel_app_v3 &"
/bin/bash -c "$work_path/double_wheel_app_v3 &"

#/bin/bash -c "source $work_path/scripts/set_env.sh && ros2 launch DEFENDER_WITHOUTARM physics.py &"
