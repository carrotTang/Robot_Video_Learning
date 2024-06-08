import pybullet as p
import pybullet_data
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])
p.setGravity(0, 0, -9.8)

# kukaUid = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
pandaUid = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
tableUid = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
trayUid = p.loadURDF("tray/tray.urdf", basePosition=[0.65, 0, 0])
objectUid = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[0.7, 0, 0.7])

state_durations = [1, 1, 1, 1]
control_dt = 1./240.
p.setTimeStep = control_dt
state_t = 0.
# 抓取4个步骤：current_state==0时，将机械臂移动到物体的上面，打开夹爪；current_state==1时，往下移动夹爪；current_state==3时，夹住物体。
current_state = 0

while True:
    state_t += control_dt
    # if you want to see the rendering immediately
    # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

    if current_state == 0:
        # 在Panda机器人的URDF定义里，joint 0~6是关节电机，9, 10是夹爪电机。
        p.setJointMotorControl2(pandaUid, 0, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL, math.pi / 4.)
        p.setJointMotorControl2(pandaUid, 2, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL, -math.pi / 2.)
        p.setJointMotorControl2(pandaUid, 4, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(pandaUid, 5, p.POSITION_CONTROL, 3 * math.pi / 4)
        p.setJointMotorControl2(pandaUid, 6, p.POSITION_CONTROL, -math.pi / 4.)
        p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, 0.08)
        p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, 0.08)
    if current_state == 1:
        p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL, math.pi / 4. + .15)
        p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL, -math.pi / 2. + .15)
    if current_state == 2:
        p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, force=200)
        p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, force=200)
    if current_state == 3:
        p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL, math.pi / 4. - 1)
        p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL, -math.pi / 2. - 1)

    if state_t > state_durations[current_state]:
        current_state += 1
        if current_state >= len(state_durations):
            current_state = 0
        state_t = 0

    print(p.getJointInfo(pandaUid,1))

    p.stepSimulation()
