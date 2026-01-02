# SynriaRD: Synria Robot Descriptions

English | [中文](README.md)

This repository contains URDF (Unified Robot Description Format) and MJCF (MuJoCo Modeling Format) models for Synria robotic platforms.

## Repository Structure

```
├── synriard
│   ├── meshes
│   │   ├── Alicia_D_v5_5
│   │   ├── Alicia_D_v5_6
│   │   ├── Alicia_M_v1_0
│   │   └── Bessica_D_v1_0
│   ├── mjcf
│   │   ├── Alicia_D_v5_5
│   │   ├── Alicia_D_v5_6
│   │   ├── Alicia_M_v1_0
│   │   └── Bessica_D_v1_0
│   └── urdf
│       ├── Alicia_D_v5_5
│       ├── Alicia_D_v5_6
│       ├── Alicia_M_v1_0
│       └── Bessica_D_v1_0
```

## Products

### Alicia-D 
- **Description**: Agile manipulation arm
- **DOF**: 6
- **Gripper Configurations**: 50mm and 100mm
- **Location**: [`Alicia_D_v5_5`](synriard/urdf/Alicia_D_v5_5) and [`Alicia_D_v5_6`](synriard/urdf/Alicia_D_v5_6)

### Alicia-M 
- **Description**: Cloud-powered robotic arm
- **DOF**: 6
- **Location**: [`Alicia_M_v1_0.urdf`](synriard/urdf/Alicia_M_v1_0/Alicia_M_v1_0.urdf)

### Bessica-D 
- **Description**: Dual-arm humanoid robot
- **DOF**: 14 (Dual 7-DOF arms)
- **Appearance**: Skeleton and covered versions

#### Skeleton Version
- **URDF**: [`Bessica_D_v1_0_Skeleton.urdf`](synriard/urdf/Bessica_D_v1_0/Bessica_D_Skeleton.urdf)

#### Covered Version
- **URDF**: [`Bessica_D_v1_0_Covered.urdf`](synriard/urdf/Bessica_D_v1_0/Bessica_D_Covered.urdf)
- **MuJoCo XML**: [`Bessica_D_v1_0_Covered.xml`](synriard/mjcf/Bessica_D_v1_0/Bessica_D_Covered.xml)

## Supported Simulation Environments

- ROS/ROS2 (via URDF)
- MuJoCo (via MJCF)
- Gazebo (via URDF)
- PyBullet (via URDF)
- Isaac Sim (via URDF/MJCF)