# Copyright 2022 The Plenoptix Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The Graph module contains all trainable parameters.
"""
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Union

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import Parameter
from torchtyping import TensorType

from nerfactory.cameras.cameras import Camera
from nerfactory.cameras.rays import RayBundle
from nerfactory.data.structs import DatasetInputs, SceneBounds
from nerfactory.graphs.modules.ray_generator import RayGenerator
from nerfactory.utils import profiler
from nerfactory.utils.callbacks import Callback
from nerfactory.utils.config import GraphConfig
from nerfactory.utils.misc import (
    get_masked_dict,
    instantiate_from_dict_config,
    is_not_none,
)


@profiler.time_function
def setup_graph(config: GraphConfig, dataset_inputs: DatasetInputs, device: str) -> "Graph":
    """Setup the graph. The dataset inputs should be set with the training data.

    Args:
        dataset_inputs: The inputs which will be used to define the camera parameters.
    """
    graph = instantiate_from_dict_config(DictConfig(config), **dataset_inputs.as_dict())
    graph.to(device)
    return graph


class AbstractGraph(nn.Module):
    """Highest level graph class. Somewhat useful to lift code up and out of the way."""

    def __init__(self) -> None:
        super().__init__()
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the graph is on."""
        return self.device_indicator_param.device

    @abstractmethod
    def forward(self, ray_indices: TensorType["num_rays", 3], batch: Union[str, Dict[str, torch.tensor]] = None):
        """Process starting with ray indices. Turns them into rays, then performs volume rendering."""


class Graph(AbstractGraph):
    """Graph class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        intrinsics: Camera intrinsics.
        camera_to_world: Camera to world transformation.
        loss_coefficients: Loss specific weights.
        scene_bounds: Bounds of target scene.
        enable_collider: Whether to create a scene collider to filter rays.
        collider_config: Configuration of scene collider.
        enable_density_field: Whether to create a density field to filter samples.
        density_field_config: Configuration of density field.
    """

    def __init__(
        self,
        intrinsics: torch.Tensor = None,
        camera_to_world: torch.Tensor = None,
        loss_coefficients: DictConfig = None,
        scene_bounds: SceneBounds = None,
        enable_collider: bool = True,
        collider_config: DictConfig = None,
        enable_density_field: bool = False,
        density_field_config: DictConfig = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert is_not_none(scene_bounds), "scene_bounds is needed to use the density grid"
        self.intrinsics = intrinsics
        self.camera_to_world = camera_to_world
        self.scene_bounds = scene_bounds
        self.enable_collider = enable_collider
        self.collider_config = collider_config
        self.loss_coefficients = loss_coefficients
        self.enable_density_field = enable_density_field
        self.density_field_config = density_field_config
        self.density_field = None
        self.kwargs = kwargs
        self.collider = None
        self.ray_generator = RayGenerator(self.intrinsics, self.camera_to_world)
        self.populate_density_field()
        self.populate_collider()
        self.populate_fields()
        self.populate_misc_modules()  # populate the modules
        self.callbacks = None
        # variable for visualizer to fetch TODO(figure out if there is cleaner way to do this)
        self.vis_outputs = None
        self.default_output_name = None

    def get_training_callbacks(self) -> List[Callback]:  # pylint:disable=no-self-use
        """Returns a list of callbacks that run functions at the specified training iterations."""
        return []

    def populate_density_field(self):
        """Set the scene density field to use."""
        if self.enable_density_field:
            self.density_field = instantiate_from_dict_config(self.density_field_config)

    def populate_collider(self):
        """Set the scene bounds collider to use."""
        if self.enable_collider:
            self.collider = instantiate_from_dict_config(self.collider_config, scene_bounds=self.scene_bounds)

    @abstractmethod
    def populate_misc_modules(self):
        """Initializes any additional modules that are part of the network."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

    @abstractmethod
    def get_outputs(self, ray_bundle: RayBundle) -> dict:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays.

        Returns:
            Outputs of graph. (ie. rendered colors)
        """

    def process_outputs_as_images(self, outputs):  # pylint:disable=no-self-use
        """Process output images into visualizable colored images"""
        # TODO override this function elsewhere or do something else for processing images
        for k, v in outputs.items():
            if v.shape[-1] == 1:
                v = torch.tile(v, (1, 1, 3))
            outputs[k] = v

    def forward_after_ray_generator(self, ray_bundle: RayBundle, batch: Union[str, Dict[str, torch.tensor]] = None):
        """Run forward starting with a ray bundle."""
        if self.collider is not None:
            intersected_ray_bundle = self.collider(ray_bundle)
            valid_mask = intersected_ray_bundle.valid_mask[..., 0]
        else:
            # NOTE(ruilongli): we don't need collider for ngp
            intersected_ray_bundle = ray_bundle
            valid_mask = None

        if batch is None:
            # during inference, keep all rays
            outputs = self.get_outputs(intersected_ray_bundle)
            return outputs

        if valid_mask is not None:
            intersected_ray_bundle = intersected_ray_bundle[valid_mask]
            # during training, keep only the rays that intersect the scene. discard the rest
            batch = get_masked_dict(batch, valid_mask)  # NOTE(ethan): this is really slow if on CPU!

        outputs = self.get_outputs(intersected_ray_bundle)
        metrics_dict = self.get_metrics_dict(outputs=outputs, batch=batch)
        loss_dict = self.get_loss_dict(
            outputs=outputs, batch=batch, metrics_dict=metrics_dict, loss_coefficients=self.loss_coefficients
        )

        # scaling losses by coefficients.
        for loss_name in loss_dict.keys():
            if loss_name in self.loss_coefficients:
                loss_dict[loss_name] *= self.loss_coefficients[loss_name]
        return outputs, loss_dict, metrics_dict

    def forward(self, ray_indices: TensorType["num_rays", 3], batch: Union[str, Dict[str, torch.tensor]] = None):
        """Run the forward starting with ray indices."""
        ray_bundle = self.ray_generator.forward(ray_indices)
        return self.forward_after_ray_generator(ray_bundle, batch=batch)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.tensor]:
        """Compute and returns metrics."""
        # pylint: disable=unused-argument
        # pylint: disable=no-self-use
        return {}

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict, loss_coefficients) -> Dict[str, torch.tensor]:
        """Computes and returns the losses dict."""

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle):
        """Takes in camera parameters and computes the output of the graph."""
        assert is_not_none(camera_ray_bundle.num_rays_per_chunk)
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs = {}
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, camera_ray_bundle.num_rays_per_chunk):
            start_idx = i
            end_idx = i + camera_ray_bundle.num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward_after_ray_generator(ray_bundle)
            for output_name, output in outputs.items():
                outputs_lists[output_name].append(output)
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)
        return outputs

    def get_outputs_for_camera(self, camera: Camera):
        """Get the graph outputs for a Camera."""
        camera_ray_bundle = camera.get_camera_ray_bundle(device=self.device)
        return self.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

    @abstractmethod
    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        """Writes the test image outputs."""

    def load_graph(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path"""
        self.load_state_dict({key.replace("module.", ""): value for key, value in loaded_state["model"].items()})