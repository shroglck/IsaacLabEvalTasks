# Adding Custom Objects to Isaac Lab

This guide explains how to replace the generic primitive objects in the evaluation tasks with custom 3D assets, such as those from [Objaverse](https://objaverse.allenai.org/) or other sources, in USD format.

## 1. Acquiring and Converting Models to USD

Isaac Sim and Isaac Lab require 3D models to be in the Universal Scene Description (USD) format.
If your models from Objaverse or other sources are in GLTF, OBJ, or FBX format, you must first convert them.

### Using Isaac Sim's Asset Converter
1. Open Isaac Sim.
2. Navigate to `Window > Extensions` and ensure the `omni.isaac.asset_importer` extension is enabled.
3. Go to `Isaac Utils > Asset Importer`.
4. Select your source file (e.g., `.glb` or `.obj`) and choose an output directory for the `.usd` file.
5. Click **Import** to generate the USD file.

## 2. Setting up Physics and Collision

A visual 3D model is not enough for robotic manipulation; the simulator needs to know its physical properties (mass, friction) and collision boundaries.

1. **Open the USD in Isaac Sim.**
2. **Add Collisions**:
   - Right-click the mesh prim in the Stage window.
   - Select `Add > Physics > Colliders Preset`.
   - For simple objects, a `Convex Hull` approximation is usually best for performance and stability. For objects with holes (like a mug handle), you may need `Convex Decomposition`.
3. **Add Rigid Body**:
   - Right-click the parent Xform of the mesh.
   - Select `Add > Physics > Rigid Body with Colliders Preset`.
4. **Set Mass**:
   - In the Property panel for the Rigid Body, find the `Physics` section.
   - Set the `Mass` property to a realistic value (in kg).
5. **Save the USD file.**

## 3. Integrating the USD into the Task Configuration

Once you have your physics-ready USD, you can swap it into the task configuration.

Locate the `base_env_cfg.py` for the task you want to modify (e.g., `source/isaaclab_eval_tasks/isaaclab_eval_tasks/tasks/manipulation/grasp_object/grasp_object_base_env_cfg.py`).

Find the `RigidObjectCfg` or `AssetBaseCfg` corresponding to the object. Update the `spawn = UsdFileCfg(...)` path to point to your new custom USD:

```python
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import isaaclab.sim as sim_utils

# In your ObjectTableSceneCfg:
custom_object = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CustomObject",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[-0.45, 0.45, 0.85],
        rot=[1, 0, 0, 0]
    ),
    spawn=UsdFileCfg(
        usd_path="path/to/your/custom_object.usd", # Update this path!
        scale=(1.0, 1.0, 1.0), # Adjust scale if your object is too large or small
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    ),
)
```

**Note on Paths**: You can place your custom USDs in a local directory or on an Omniverse Nucleus server. If using a local path, ensure it is an absolute path or a path relative to your script execution directory.

## 4. Tuning Grasping Parameters

Custom objects might have different dimensions than the primitive shapes. You may need to adjust:
- The initial `pos` (position) of the object to ensure it rests correctly on the table and doesn't intersect with it.
- The success thresholds in the task's MDP functions (e.g., the minimum height to consider a successful grasp, or the minimum distance between the object and the target).
