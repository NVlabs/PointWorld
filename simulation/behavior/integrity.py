#!/usr/bin/env python3

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
"""Behavior-specific integrity helpers."""

from typing import Any, Dict, List

import numpy as np

def should_skip_behavior_clip(clip_group):
    """
    Helper function to determine if a Behavior clip should be skipped based on attributes.
    
    Skip a clip if any of the following conditions are met:
    - has trunk arm collision
    - number of scene points is less than 3000
    
    Args:
        clip_group: h5py group containing clip data with attributes
        
    Returns:
        tuple: (should_skip: bool, skip_reason: str or None)
    """
    required_attrs = ['has_trunk_arm_collision', 'num_scene_points']
    for attr in required_attrs:
        assert attr in clip_group.attrs, f"Required attribute '{attr}' not found in clip"
    
    # Check trunk arm collision
    if clip_group.attrs['has_trunk_arm_collision']:
        return True, 'trunk_arm_collision'
    
    # Check scene point count
    if clip_group.attrs['num_scene_points'] < 3000:
        return True, 'low_scene_points'
    
    return False, None

def analyze_behavior_gripper_filtering(clip_group):
    """
    Analyze Behavior gripper filtering conditions for a clip.
    
    Based on dataset.py logic:
    - Skip a gripper if: no collision AND min distance > 20cm (0.2m)
    
    Args:
        clip_group: h5py group containing clip data with attributes
        
    Returns:
        dict: Analysis results with filtering statistics
    """
    required_attrs = [
        'has_left_gripper_finger_collision', 'left_min_distance_to_all_objects',
        'has_right_gripper_finger_collision', 'right_min_distance_to_all_objects'
    ]
    
    analysis = {
        'has_required_attrs': True,
        'left_gripper_skip_eligible': False,
        'right_gripper_skip_eligible': False,
        'both_grippers_skip_eligible': False,
        'left_no_collision': False,
        'left_far_distance': False,
        'right_no_collision': False,
        'right_far_distance': False,
        'left_min_distance': None,
        'right_min_distance': None
    }
    
    # Check if all required attributes are present
    for attr in required_attrs:
        if attr not in clip_group.attrs:
            analysis['has_required_attrs'] = False
            return analysis
    
    # Extract attribute values and convert to Python native types
    left_has_collision = clip_group.attrs['has_left_gripper_finger_collision']
    left_min_distance = clip_group.attrs['left_min_distance_to_all_objects']
    right_has_collision = clip_group.attrs['has_right_gripper_finger_collision']
    right_min_distance = clip_group.attrs['right_min_distance_to_all_objects']
    
    # Convert to Python native types for JSON serialization
    if hasattr(left_has_collision, 'item'):
        left_has_collision = left_has_collision.item()
    if hasattr(left_min_distance, 'item'):
        left_min_distance = left_min_distance.item()
    if hasattr(right_has_collision, 'item'):
        right_has_collision = right_has_collision.item()
    if hasattr(right_min_distance, 'item'):
        right_min_distance = right_min_distance.item()
    
    # Store distance values for statistics
    analysis['left_min_distance'] = float(left_min_distance)
    analysis['right_min_distance'] = float(right_min_distance)
    
    # Analyze left gripper conditions
    analysis['left_no_collision'] = bool(not left_has_collision)
    analysis['left_far_distance'] = bool(left_min_distance > 0.2)
    analysis['left_gripper_skip_eligible'] = bool(analysis['left_no_collision'] and analysis['left_far_distance'])
    
    # Analyze right gripper conditions
    analysis['right_no_collision'] = bool(not right_has_collision)
    analysis['right_far_distance'] = bool(right_min_distance > 0.2)
    analysis['right_gripper_skip_eligible'] = bool(analysis['right_no_collision'] and analysis['right_far_distance'])
    
    # Both grippers skip eligible
    analysis['both_grippers_skip_eligible'] = bool(analysis['left_gripper_skip_eligible'] and analysis['right_gripper_skip_eligible'])
    
    return analysis


def calculate_behavior_attribute_stats(all_clip_attributes: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate min, max, mean, std for all non-string attributes across all Behavior clips.
    
    Args:
        all_clip_attributes: List of clip attribute dictionaries
        
    Returns:
        Dictionary with attribute statistics
    """
    import numpy as np
    
    if not all_clip_attributes:
        return {}
    
    # Collect all numeric attributes
    numeric_attrs = {}
    for clip_attrs in all_clip_attributes:
        for attr_name, attr_value in clip_attrs.items():
            if isinstance(attr_value, (int, float, bool)) and not isinstance(attr_value, str):
                if attr_name not in numeric_attrs:
                    numeric_attrs[attr_name] = []
                numeric_attrs[attr_name].append(float(attr_value))
    
    # Calculate statistics
    stats = {}
    for attr_name, values in numeric_attrs.items():
        values_array = np.array(values)
        stats[attr_name] = {
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)), 
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'count': len(values)
        }
    
    return stats

def calculate_behavior_gripper_filtering_stats(all_gripper_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate Behavior gripper filtering statistics from all clip analyses.
    
    Args:
        all_gripper_analysis: List of gripper analysis dictionaries from all clips
        
    Returns:
        Dictionary with gripper filtering statistics
    """
    import numpy as np
    
    if not all_gripper_analysis:
        return {}
    
    # Filter out clips without required attributes
    valid_analyses = [analysis for analysis in all_gripper_analysis if analysis.get('has_required_attrs', False)]
    
    if not valid_analyses:
        return {'error': 'No clips with valid gripper attributes found'}
    
    total_clips = len(valid_analyses)
    
    # Count individual conditions
    left_no_collision_count = sum(1 for a in valid_analyses if a.get('left_no_collision', False))
    left_far_distance_count = sum(1 for a in valid_analyses if a.get('left_far_distance', False))
    left_skip_eligible_count = sum(1 for a in valid_analyses if a.get('left_gripper_skip_eligible', False))
    
    right_no_collision_count = sum(1 for a in valid_analyses if a.get('right_no_collision', False))
    right_far_distance_count = sum(1 for a in valid_analyses if a.get('right_far_distance', False))
    right_skip_eligible_count = sum(1 for a in valid_analyses if a.get('right_gripper_skip_eligible', False))
    
    both_skip_eligible_count = sum(1 for a in valid_analyses if a.get('both_grippers_skip_eligible', False))
    
    # Calculate distance statistics
    left_distances = [a['left_min_distance'] for a in valid_analyses if a.get('left_min_distance') is not None]
    right_distances = [a['right_min_distance'] for a in valid_analyses if a.get('right_min_distance') is not None]
    
    stats = {
        'total_clips_analyzed': total_clips,
        'individual_conditions': {
            'left_no_collision': {
                'count': left_no_collision_count,
                'percentage': (left_no_collision_count / total_clips) * 100 if total_clips > 0 else 0
            },
            'left_far_distance': {
                'count': left_far_distance_count,
                'percentage': (left_far_distance_count / total_clips) * 100 if total_clips > 0 else 0
            },
            'right_no_collision': {
                'count': right_no_collision_count,
                'percentage': (right_no_collision_count / total_clips) * 100 if total_clips > 0 else 0
            },
            'right_far_distance': {
                'count': right_far_distance_count,
                'percentage': (right_far_distance_count / total_clips) * 100 if total_clips > 0 else 0
            }
        },
        'combined_conditions': {
            'left_gripper_skip_eligible': {
                'count': left_skip_eligible_count,
                'percentage': (left_skip_eligible_count / total_clips) * 100 if total_clips > 0 else 0
            },
            'right_gripper_skip_eligible': {
                'count': right_skip_eligible_count,
                'percentage': (right_skip_eligible_count / total_clips) * 100 if total_clips > 0 else 0
            },
            'both_grippers_skip_eligible': {
                'count': both_skip_eligible_count,
                'percentage': (both_skip_eligible_count / total_clips) * 100 if total_clips > 0 else 0
            }
        }
    }
    
    # Add distance statistics
    if left_distances:
        left_distances = np.array(left_distances)
        stats['distance_statistics'] = {
            'left_gripper_distances': {
                'min': float(np.min(left_distances)),
                'max': float(np.max(left_distances)),
                'mean': float(np.mean(left_distances)),
                'std': float(np.std(left_distances)),
                'threshold_0_2': sum(1 for d in left_distances if d > 0.2),
                'threshold_0_2_percentage': (sum(1 for d in left_distances if d > 0.2) / len(left_distances)) * 100
            }
        }
    
    if right_distances:
        right_distances = np.array(right_distances)
        if 'distance_statistics' not in stats:
            stats['distance_statistics'] = {}
        stats['distance_statistics']['right_gripper_distances'] = {
            'min': float(np.min(right_distances)),
            'max': float(np.max(right_distances)),
            'mean': float(np.mean(right_distances)),
            'std': float(np.std(right_distances)),
            'threshold_0_2': sum(1 for d in right_distances if d > 0.2),
            'threshold_0_2_percentage': (sum(1 for d in right_distances if d > 0.2) / len(right_distances)) * 100
        }
    
    return stats
