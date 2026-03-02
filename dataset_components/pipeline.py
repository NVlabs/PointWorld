# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import webdataset as wds

from dataset_components.cameras import convert_to_tensors
from dataset_components.constants import (
    RELEASE_CHROMATIC_AUTO_CONTRAST_BLEND,
    RELEASE_CHROMATIC_AUTO_CONTRAST_PROB,
    RELEASE_CHROMATIC_JITTER_PROB,
    RELEASE_CHROMATIC_JITTER_STD,
    RELEASE_CHROMATIC_TRANSLATION_PROB,
    RELEASE_CHROMATIC_TRANSLATION_RATIO,
    RELEASE_CONTEXT_HORIZON,
    RELEASE_MAX_NUM_SPHERES,
    RELEASE_MAX_RELATIVE_MOVEMENT,
    RELEASE_NUM_CANDIDATES,
    RELEASE_RANDOM_CONTEXT_MODE,
    RELEASE_RANDOM_FLIP_PROB,
    RELEASE_RANDOM_SCALE_MAX,
    RELEASE_RANDOM_SCALE_MIN,
    RELEASE_SPHERECROP_BUFFER,
    RELEASE_SPHERECROP_MAX_R,
    RELEASE_SPHERECROP_MIN_R,
    RELEASE_SPHERECROP_PROB,
)
from dataset_components.robot import gather_features
from dataset_components.transforms import (
    assert_camera_payload_resolution,
    center_shift,
    chromatic_auto_contrast_transform,
    chromatic_jitter_transform,
    chromatic_translation_transform,
    compute_helper_variables,
    enforce_max_num_points,
    filter_within_bounds,
    grid_sample_transform,
    make_gt_copy,
    normalize_colors,
    random_flip_transform,
    random_rotate_around_z_axis,
    random_scale_transform,
    sample_and_apply_scene_context_mask,
    sphere_crop_transform,
)


def gather(sample, fields, domain):
    result = {}

    # Handle non-image fields
    for field in fields:
        if field != '__domain__' and field in sample:
            result[field] = sample[field]
        elif field == '__domain__':
            result['__domain__'] = domain

    # Handle image fields based on expected suffix patterns.
    image_field_patterns = ['_initial_rgb', '_initial_depth', '_intrinsic', '_extrinsic']
    for pattern in image_field_patterns:
        if pattern in fields:
            # Find all keys in sample that end with this pattern
            for key in sample.keys():
                if key.endswith(pattern):
                    result[key] = sample[key]

    return result


def sample_transform_pipeline(dataset, data_dir, domain, mode, args, has_bimanual_robot=False, rank: int = 0):
    """
    Sample transformations pipeline.
    """
    # -------------------------------------
    # shared sample-specific transformations across all modes (train, test)
    # -------------------------------------
    dataset = (
        dataset
        .map(center_shift)
        .map(filter_within_bounds)
        .map(partial(assert_camera_payload_resolution, expected_hw=(180, 320)))
    )
    # -------------------------------------
    # mode-specific sample-specific transformations
    # -------------------------------------
    if mode == 'train':
        if args.deterministic_train:
            dataset = (
                dataset
                .map(partial(grid_sample_transform, grid_size=args.grid_size, mode=mode))
                .map(partial(enforce_max_num_points,
                    max_scene_points=args.max_scene_points,
                    deterministic=True, seed=(args.seed + rank)),
                    handler=wds.warn_and_continue)
                .map(center_shift)
                .map(normalize_colors)
                .map(make_gt_copy)
                .map(partial(sample_and_apply_scene_context_mask, random_context_mode=RELEASE_RANDOM_CONTEXT_MODE, context_horizon=RELEASE_CONTEXT_HORIZON))
            )
        else:
            dataset = (
                dataset
                .map(partial(grid_sample_transform, grid_size=args.grid_size, mode=mode))
                .map(partial(sphere_crop_transform,
                    prob=RELEASE_SPHERECROP_PROB,
                    min_radius=RELEASE_SPHERECROP_MIN_R,
                    max_radius=RELEASE_SPHERECROP_MAX_R,
                    buffer=RELEASE_SPHERECROP_BUFFER,
                    num_candidates=RELEASE_NUM_CANDIDATES,
                    max_num_spheres=RELEASE_MAX_NUM_SPHERES,
                    max_scene_points=args.max_scene_points),
                    handler=wds.warn_and_continue)
                .map(partial(enforce_max_num_points,
                    max_scene_points=args.max_scene_points))
                .map(random_rotate_around_z_axis)
                .map(partial(random_scale_transform,
                            scale_range=[RELEASE_RANDOM_SCALE_MIN, RELEASE_RANDOM_SCALE_MAX]))
                .map(partial(random_flip_transform,
                            p=RELEASE_RANDOM_FLIP_PROB))
                .map(center_shift)
                .map(partial(chromatic_auto_contrast_transform,
                            p=RELEASE_CHROMATIC_AUTO_CONTRAST_PROB,
                            blend_factor=RELEASE_CHROMATIC_AUTO_CONTRAST_BLEND))
                .map(partial(chromatic_translation_transform,
                            p=RELEASE_CHROMATIC_TRANSLATION_PROB,
                            ratio=RELEASE_CHROMATIC_TRANSLATION_RATIO))
                .map(partial(chromatic_jitter_transform,
                            p=RELEASE_CHROMATIC_JITTER_PROB,
                            std=RELEASE_CHROMATIC_JITTER_STD))
                .map(normalize_colors)
                .map(make_gt_copy)
                .map(partial(sample_and_apply_scene_context_mask, random_context_mode=RELEASE_RANDOM_CONTEXT_MODE, context_horizon=RELEASE_CONTEXT_HORIZON))
            )
    elif mode == 'test':
        dataset = (
            dataset
            .map(partial(grid_sample_transform, grid_size=args.grid_size, mode=mode))
            .map(partial(enforce_max_num_points,
                max_scene_points=args.max_scene_points,
                deterministic=True, seed=(args.seed + rank)),
                handler=wds.warn_and_continue)
            .map(center_shift)
            .map(normalize_colors)
            .map(make_gt_copy)
            .map(partial(sample_and_apply_scene_context_mask, random_context_mode=RELEASE_RANDOM_CONTEXT_MODE, context_horizon=RELEASE_CONTEXT_HORIZON))
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # -------------------------------------
    # final transformations
    # -------------------------------------
    final_fields = [
        'scene_flows',
        'scene_features',
        'scene_visibility',
        'scene_depth_valid_mask',
        'robot_flows',
        'robot_features',
        'joint_positions',
        'gripper_positions',
        'gt_scene_flows',
        'scene_context_mask',
        '__key__',
        '__domain__',
        '__out_of_bounds__',
        '__scene_exceeds_max__',
        '__world_transform__',
        '__world_reflection__',
    ]

    # Only include both sides if any dataset in this run is bimanual (e.g., behavior in args.domains)
    if has_bimanual_robot:
        final_fields += [
            'right_gripper_pose', 'left_gripper_pose', 'right_gripper_open', 'left_gripper_open',
            'joint_names', 'base_pose',
        ]
    else:
        final_fields += ['right_gripper_pose', 'right_gripper_open']

    # Always include image and camera matrix fields for release.
    # These fields will have camera prefixes like camera_0_initial_rgb, camera_1_initial_depth, etc.
    # We use a pattern to match any camera prefix.
    image_fields = ['_initial_rgb', '_initial_depth', '_intrinsic', '_extrinsic']
    final_fields.extend(image_fields)

    if mode == 'test' and '__shift_amount__' not in final_fields:
        final_fields.append('__shift_amount__')

    dataset = (
        dataset
        .map(partial(gather_features,
                     robot_features=args.robot_features,
                     scene_features=args.scene_features,
                    random_context_mode=RELEASE_RANDOM_CONTEXT_MODE,
                    context_horizon=RELEASE_CONTEXT_HORIZON,
                    has_bimanual_robot=has_bimanual_robot,
                    domain=domain), handler=wds.warn_and_continue)
        .map(partial(gather, fields=final_fields, domain=domain))
        .map(partial(compute_helper_variables, max_relative_movement=RELEASE_MAX_RELATIVE_MOVEMENT, domain=domain))
        .map(convert_to_tensors)
    )
    return dataset


def apply_release_pipeline_to_sample(
    sample: dict,
    domain: str,
    mode: str,
    args,
    *,
    has_bimanual_robot: bool = False,
    rank: int = 0,
    include_scene_data: bool = False,
    skip_scene_sampling: bool = False,
):
    """Apply the release transform pipeline to a single in-memory sample dict."""
    if mode not in ("train", "test"):
        raise ValueError(f"Invalid mode: {mode}")

    # -------------------------------------
    # shared sample-specific transformations across all modes (train, test)
    # -------------------------------------
    sample = center_shift(sample)
    if skip_scene_sampling:
        # Teleop uses pre-filtered scenes; keep counts stable.
        sample.setdefault('__out_of_bounds__', False)
    else:
        sample = filter_within_bounds(sample)

    # Fail-fast: camera payloads must already match the release contract resolution.
    sample = assert_camera_payload_resolution(sample, expected_hw=(180, 320))

    # -------------------------------------
    # mode-specific sample-specific transformations
    # -------------------------------------
    if mode == 'train':
        if args.deterministic_train:
            sample = grid_sample_transform(sample, grid_size=args.grid_size, mode=mode)
            sample = enforce_max_num_points(
                sample,
                max_scene_points=args.max_scene_points,
                deterministic=True,
                seed=(args.seed + rank),
            )
            sample = center_shift(sample)
            sample = normalize_colors(sample)
            sample = make_gt_copy(sample)
            sample = sample_and_apply_scene_context_mask(
                sample,
                random_context_mode=RELEASE_RANDOM_CONTEXT_MODE,
                context_horizon=RELEASE_CONTEXT_HORIZON,
            )
        else:
            sample = grid_sample_transform(sample, grid_size=args.grid_size, mode=mode)
            sample = sphere_crop_transform(
                sample,
                prob=RELEASE_SPHERECROP_PROB,
                min_radius=RELEASE_SPHERECROP_MIN_R,
                max_radius=RELEASE_SPHERECROP_MAX_R,
                buffer=RELEASE_SPHERECROP_BUFFER,
                num_candidates=RELEASE_NUM_CANDIDATES,
                max_num_spheres=RELEASE_MAX_NUM_SPHERES,
                max_scene_points=args.max_scene_points,
            )
            sample = enforce_max_num_points(sample, max_scene_points=args.max_scene_points)
            sample = random_rotate_around_z_axis(sample)
            sample = random_scale_transform(sample, scale_range=[RELEASE_RANDOM_SCALE_MIN, RELEASE_RANDOM_SCALE_MAX])
            sample = random_flip_transform(sample, p=RELEASE_RANDOM_FLIP_PROB)
            sample = center_shift(sample)
            sample = chromatic_auto_contrast_transform(
                sample,
                p=RELEASE_CHROMATIC_AUTO_CONTRAST_PROB,
                blend_factor=RELEASE_CHROMATIC_AUTO_CONTRAST_BLEND,
            )
            sample = chromatic_translation_transform(
                sample,
                p=RELEASE_CHROMATIC_TRANSLATION_PROB,
                ratio=RELEASE_CHROMATIC_TRANSLATION_RATIO,
            )
            sample = chromatic_jitter_transform(
                sample,
                p=RELEASE_CHROMATIC_JITTER_PROB,
                std=RELEASE_CHROMATIC_JITTER_STD,
            )
            sample = normalize_colors(sample)
            sample = make_gt_copy(sample)
            sample = sample_and_apply_scene_context_mask(
                sample,
                random_context_mode=RELEASE_RANDOM_CONTEXT_MODE,
                context_horizon=RELEASE_CONTEXT_HORIZON,
            )
    elif mode == 'test':
        if not skip_scene_sampling:
            sample = grid_sample_transform(sample, grid_size=args.grid_size, mode=mode)
            sample = enforce_max_num_points(
                sample,
                max_scene_points=args.max_scene_points,
                deterministic=True,
                seed=(args.seed + rank),
            )
        else:
            sample.setdefault('__scene_exceeds_max__', False)
        sample = center_shift(sample)
        sample = normalize_colors(sample)
        sample = make_gt_copy(sample)
        sample = sample_and_apply_scene_context_mask(
            sample,
            random_context_mode=RELEASE_RANDOM_CONTEXT_MODE,
            context_horizon=RELEASE_CONTEXT_HORIZON,
        )

    # -------------------------------------
    # final transformations
    # -------------------------------------
    final_fields = [
        'scene_flows',
        'scene_features',
        'scene_visibility',
        'scene_depth_valid_mask',
        'robot_flows',
        'robot_features',
        'joint_positions',
        'gripper_positions',
        'gt_scene_flows',
        'scene_context_mask',
        '__key__',
        '__domain__',
        '__out_of_bounds__',
        '__scene_exceeds_max__',
        '__world_transform__',
        '__world_reflection__',
    ]

    # Only include both sides if any dataset in this run is bimanual (e.g., behavior in args.domains)
    if has_bimanual_robot:
        final_fields += [
            'right_gripper_pose', 'left_gripper_pose', 'right_gripper_open', 'left_gripper_open',
            'joint_names', 'base_pose',
        ]
    else:
        final_fields += ['right_gripper_pose', 'right_gripper_open']

    # Always include image and camera matrix fields for release.
    image_fields = ['_initial_rgb', '_initial_depth', '_intrinsic', '_extrinsic']
    final_fields.extend(image_fields)

    if include_scene_data:
        final_fields += ['scene_colors', 'scene_normals']

    if mode == 'test' and '__shift_amount__' not in final_fields:
        final_fields.append('__shift_amount__')

    sample = gather_features(
        sample,
        robot_features=args.robot_features,
        scene_features=args.scene_features,
        random_context_mode=RELEASE_RANDOM_CONTEXT_MODE,
        context_horizon=RELEASE_CONTEXT_HORIZON,
        has_bimanual_robot=has_bimanual_robot,
        domain=domain,
    )
    sample = gather(sample, fields=final_fields, domain=domain)
    sample = compute_helper_variables(sample, max_relative_movement=RELEASE_MAX_RELATIVE_MOVEMENT, domain=domain)
    sample = convert_to_tensors(sample)
    return sample
