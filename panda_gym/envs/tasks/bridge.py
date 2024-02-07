from typing import Any, Dict, Tuple

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Bridge(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.1,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.06
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=2.0, width=1.2, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="block_1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.4, -0.3, self.object_size / 2]),
            orientation=np.array([1.,1.,0.,1.]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        #self.sim.create_box(
        #    body_name="target_1",
        #    half_extents=np.ones(3) * self.object_size / 2,
        #    mass=0.0,
        #    ghost=True,
        #    position=np.array([-2*self.object_size+0.5, 0.3, self.object_size / 2]),
        #    rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        #)
        self.sim.create_box(
            body_name="block_2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.6, 0.2, self.object_size / 2]),
            orientation=np.array([1.,1.,0.,1.]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        """ self.sim.create_box(
            body_name="target_2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.3, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        ) """
        self.sim.create_box(
            body_name="block_3",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.6, 0.4, self.object_size / 2]),
            orientation=np.array([1.,1.,0.,1.]),
            rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
        )
        """ self.sim.create_box(
            body_name="target3",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([2*self.object_size+0.5, 0.3, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
        ) """

        self.sim.create_box(
            body_name="block_4",
            half_extents=np.array([2,1,0.8]) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.6, 0.0, 0.8*self.object_size / 2]),
            orientation=np.array([1.,1.,0.,1.]),
            rgba_color=np.array([0.2, 0.3, 0.1, 1.0]),
        )
        """ self.sim.create_box(
            body_name="target4",
            half_extents=np.array([2,1,0.5])* self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([-self.object_size+0.5, 0.3, 1.25*self.object_size]),
            rgba_color=np.array([0.2, 0.3, 0.1, 0.3]),
        ) """

        self.sim.create_box(
            body_name="block_5",
            half_extents=np.array([2,1,0.8]) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.6, -0.3, 0.8*self.object_size / 2]),
            orientation=np.array([1.,1.,0.,1.]),
            rgba_color=np.array([0.7, 0.6, 0.1, 1.0]),
        )
        """ self.sim.create_box(
            body_name="target5",
            half_extents=np.array([2,1,0.5])* self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([self.object_size+0.5, 0.3, 1.25*self.object_size]),
            rgba_color=np.array([0.7, 0.6, 0.1, 0.3]),
        ) """

    def get_obs(self) -> np.ndarray:
        # position of objects
        obs = {
            "block_1": np.array(self.sim.get_base_position("block_1")),
            "block_2": np.array(self.sim.get_base_position("block_2")),
            "block_3": np.array(self.sim.get_base_position("block_3")),
            "block_4": np.array(self.sim.get_base_position("block_4")),
            "block_5": np.array(self.sim.get_base_position("block_5")),
            "block_1_rotation": np.array(self.sim.get_base_rotation("block_1","euler")),
            "block_2_rotation": np.array(self.sim.get_base_rotation("block_2","euler")),
            "block_3_rotation": np.array(self.sim.get_base_rotation("block_3","euler")),
            "block_4_rotation": np.array(self.sim.get_base_rotation("block_4","euler")),
            "block_5_rotation": np.array(self.sim.get_base_rotation("block_5","euler"))
        }
        return obs

    def get_achieved_goal(self) -> np.ndarray:
        return np.zeros(1)

    def reset(self) -> None:
        self.goal = self._sample_goal()
        block_1_position, block_2_position, block_3_position, block_4_position, block_5_position = self._sample_objects()
        self.sim.set_base_pose("block_1", block_1_position, np.array([0.,0.,0.,1.]))
        self.sim.set_base_pose("block_2", block_2_position, np.array([0.,0.,0.,1.]))
        self.sim.set_base_pose("block_3", block_3_position, np.array([0.,0.,0.,1.]))
        self.sim.set_base_pose("block_4", block_4_position, np.array([0.,0.,0.,1.]))
        self.sim.set_base_pose("block_5", block_5_position, np.array([0.,0.,0.,1.]))
    
    def _sample_goal(self) -> np.ndarray:
        return np.zeros(1)

    

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        block_1_position = np.array([0.4, -0.3, self.object_size / 2])
        block_2_position = np.array([0.6, 0.2, self.object_size / 2])
        block_3_position = np.array([0.6, 0.4, self.object_size / 2])
        block_4_position = np.array([0.6, 0.0, 0.8*self.object_size / 2])
        block_5_position = np.array([0.6, -0.3, 0.8*self.object_size / 2])
        return block_1_position, block_2_position, block_3_position, block_4_position, block_5_position


    def _get_object_orietation(self):
        block1_rotation=np.array(self.sim.get_base_rotation("block_1","quaternion"))
        block2_rotation=np.array(self.sim.get_base_rotation("block_2","quaternion"))
        block3_rotation=np.array(self.sim.get_base_rotation("block_3","quaternion"))
        block4_rotation=np.array(self.sim.get_base_rotation("block_4","quaternion"))
        block5_rotation=np.array(self.sim.get_base_rotation("block_5","quaternion"))
        return block1_rotation,block2_rotation, block3_rotation, block4_rotation, block5_rotation

    def is_success(self):
        # harcoded to False
        return False
    
    def compute_cost(self):
        # hardcoded to 0
        return 0

    def compute_reward(self):
        # harcoded to 0
        return 0
