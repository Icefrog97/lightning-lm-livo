# calc_imu_tilt.py

静止录制 IMU bag，估计相对地面的 Roll / Pitch，用于填写 `common/E_body_wrt_lio`（与 `common/T_body_wrt_lio` 一起组成车体相对 LIO 的调平外参，`[yaw, pitch, roll]` 度）。

```bash
python3 calc_imu_tilt.py <bag文件> <imu_topic>
# 例: python3 calc_imu_tilt.py imu.bag /livox/imu
```

将输出的 Pitch、Roll 写入：`E_body_wrt_lio: [<yaw>, <Pitch>, <Roll>]`。车辆需停在水平地面上录包。
