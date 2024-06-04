import pybullet as p
import pybullet_data
import time

# 启动仿真引擎的GUI
p.connect(p.GUI)

# 设置重力加速度
p.setGravity(0, 0, -9.81)

# 加载URDF模型路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载平面模型作为地面
planeId = p.loadURDF("plane.urdf")

# 设置两个球体的初始位置
ball1StartPos = [-1, 0, 0.5]
ball2StartPos = [1, 0, 0.5]

# 加载第一个球体模型
ball1Id = p.loadURDF("sphere2.urdf", ball1StartPos)

# 加载第二个球体模型
ball2Id = p.loadURDF("sphere2.urdf", ball2StartPos)

# 设置初始速度，使两个球体朝对方运动
p.resetBaseVelocity(ball1Id, linearVelocity=[5, 0, 0])
p.resetBaseVelocity(ball2Id, linearVelocity=[-5, 0, 0])

# 设置模拟循环和时间步长
timeStep = 1./240.
p.setTimeStep(timeStep)

# 模拟循环，持续一定时间
for i in range(500):
    p.stepSimulation()
    time.sleep(timeStep)

# 断开与仿真引擎的连接
p.disconnect()
