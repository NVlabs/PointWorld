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
import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
import time
from typing import List, Dict, Tuple, Any
import multiprocessing as mp
from collections import Counter
import json
from simulation.behavior.integrity import (
    analyze_behavior_gripper_filtering,
    calculate_behavior_attribute_stats,
    calculate_behavior_gripper_filtering_stats,
    should_skip_behavior_clip,
)
from shared.data_contract import (
    BEHAVIOR_CLIP_ATTRIBUTE_KEYS,
    EXPECTED_CAMERA_PAYLOAD_SHAPES,
    IMAGE_KEYS,
    get_wds_data_keys,
    validate_domain,
)

class IntegrityError(Exception):
    """Exception raised when an integrity issue is found in debug mode."""
    pass


QUANTIZED_NORMALS_DTYPE = np.int8


def _is_normals_modality(modality: str) -> bool:
    return modality in {"scene_normals", "local_scene_normals"}


def _check_quantized_normals_dataset_dtype(payload: h5py.Dataset, context: str) -> Tuple[bool, str]:
    if not isinstance(payload, h5py.Dataset):
        return False, f"{context} is not a dataset"
    if np.dtype(payload.dtype) != np.dtype(QUANTIZED_NORMALS_DTYPE):
        return False, (
            f"{context} expected dtype {QUANTIZED_NORMALS_DTYPE}, "
            f"got {payload.dtype}"
        )
    return True, ""


def check_normals_payload_contract(
    camera_data: h5py.Group,
    modality: str,
    h5_path: str,
    clip_key: str,
    camera_key: str,
) -> Tuple[bool, str]:
    if modality not in camera_data:
        return False, f"missing {modality}"
    payload = camera_data[modality]
    context = f"{h5_path}/{clip_key}/{camera_key}/{modality}"
    if isinstance(payload, h5py.Dataset):
        return _check_quantized_normals_dataset_dtype(payload, context)
    if isinstance(payload, h5py.Group):
        for child_key in payload.keys():
            child = payload[child_key]
            child_context = f"{context}/{child_key}"
            if not isinstance(child, h5py.Dataset):
                return False, f"{child_context} is not a dataset"
            is_valid, reason = _check_quantized_normals_dataset_dtype(child, child_context)
            if not is_valid:
                return False, reason
        return True, ""
    return False, f"{context} has invalid container type {type(payload)}"


def check_clip_normals_contract(
    clip_data: h5py.Group,
    data_keys: List[str],
    h5_path: str,
    clip_key: str,
) -> List[str]:
    issues: List[str] = []
    camera_keys = [k for k in clip_data.keys() if k.startswith("camera_")]
    if not camera_keys:
        return ["No valid camera data"]
    normals_modalities = [m for m in data_keys if _is_normals_modality(m)]
    for camera_key in camera_keys:
        camera_data = clip_data[camera_key]
        if not isinstance(camera_data, h5py.Group):
            issues.append(
                f"Invalid camera container at {h5_path}/{clip_key}/{camera_key}: {type(camera_data)}"
            )
            continue
        for modality in normals_modalities:
            is_valid, reason = check_normals_payload_contract(
                camera_data,
                modality=modality,
                h5_path=h5_path,
                clip_key=clip_key,
                camera_key=camera_key,
            )
            if not is_valid:
                issues.append(reason)
    return issues


def check_exists_and_valid(group, field, check_write_complete=False):
    """Check if a field exists in the h5py group and has valid data."""
    if field not in group:
        return False
    
    try:
        data = group[field]
        if isinstance(data, h5py.Dataset):
            # Check if data has valid shape
            if data.shape is None or 0 in data.shape:
                return False
            # Try reading a small slice to check validity
            if len(data.shape) > 0:
                data[0:min(1, data.shape[0])]
            
            # Check for write_complete attribute if requested
            if check_write_complete:
                if 'write_complete' not in data.attrs or not data.attrs['write_complete']:
                    return False
            
            return True
        elif isinstance(data, h5py.Group):
            return True
    except Exception:
        return False
    
    return False

def check_scene_modality_dimensions(camera_data, scene_modalities, check_write_complete=False):
    """
    Check that all scene modalities have consistent first two dimensions (T, NS).
    
    Args:
        camera_data: h5py group containing camera data
        scene_modalities: list of scene modality names to check
        check_write_complete: whether to check for write_complete attribute
        
    Returns:
        tuple: (is_consistent, expected_shape, mismatched_modalities)
    """
    expected_shape = None
    mismatched_modalities = []
    
    for modality in scene_modalities:
        if check_exists_and_valid(camera_data, modality, check_write_complete):
            try:
                data = camera_data[modality]
                if isinstance(data, h5py.Dataset) and len(data.shape) >= 2:
                    current_shape = data.shape[:2]  # First two dimensions (T, NS)
                    
                    if expected_shape is None:
                        expected_shape = current_shape
                    elif current_shape != expected_shape:
                        mismatched_modalities.append(f"{modality}: {current_shape}")
            except Exception:
                # If we can't read the data, mark it as mismatched
                mismatched_modalities.append(f"{modality}: unreadable")
    
    is_consistent = len(mismatched_modalities) == 0
    return is_consistent, expected_shape, mismatched_modalities


def check_camera_payload_contract(camera_data, modality: str):
    if modality not in camera_data:
        return False, f"missing {modality}"
    payload = camera_data[modality]
    if not isinstance(payload, h5py.Dataset):
        return False, f"{modality} is not a dataset"

    if modality == "initial_rgb":
        if payload.shape != (1,):
            return False, f"initial_rgb expected shape (1,), got {payload.shape}"
        return True, ""

    expected_shape = EXPECTED_CAMERA_PAYLOAD_SHAPES[modality]
    if tuple(payload.shape) != expected_shape:
        return False, f"{modality} expected shape {expected_shape}, got {tuple(payload.shape)}"
    return True, ""


def check_file_integrity(h5_path: str, data_keys: List[str], debug: bool = False, fastmode: bool = False, check_write_complete: bool = False, domain: str = None) -> Dict[str, Any]:
    """Check integrity of a single h5 file."""
    validate_domain(domain)
    results = {
        'file_path': h5_path,
        'status': 'failed',
        'total_clips': 0,
        'valid_clips': 0,
        'invalid_clips': 0,
        'skipped_clips': 0,
        'skip_reasons': {},  # Track reasons for skipping clips
        'clip_details': [],
        'clip_attributes': [],  # For Behavior domain attribute collection
        'gripper_analysis': []  # For Behavior domain gripper filtering analysis
    }
    
    try:
        with h5py.File(h5_path, "r") as f:
            uuid = f.attrs.get('uuid', 'unknown')
            results['uuid'] = uuid
            
            # Speedup: if episode_complete flag is present and true, use clip_complete flags for validation
            episode_complete = f.attrs.get('episode_complete', False)
            use_speedup = episode_complete
            
            for clip_key, clip_data in f.items():
                if not ':' in clip_key:
                    continue
                
                clip_result = {
                    'clip_key': clip_key,
                    'status': 'invalid',
                    'issues': []
                }
                results['total_clips'] += 1
                
                # Behavior domain - collect attributes and check if clip should be skipped BEFORE speedup mode
                if domain == 'behavior':
                    missing_behavior_attrs = [
                        attr_name
                        for attr_name in BEHAVIOR_CLIP_ATTRIBUTE_KEYS
                        if attr_name not in clip_data.attrs
                    ]
                    if missing_behavior_attrs:
                        issue = f"Missing required behavior clip attributes: {missing_behavior_attrs}"
                        clip_result['issues'].append(issue)
                        results['clip_details'].append(clip_result)
                        results['invalid_clips'] += 1
                        if debug:
                            raise IntegrityError(f"File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                        print(f"INTEGRITY ISSUE - File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                        continue

                    # Collect all clip attributes for summary statistics
                    clip_attrs = {}
                    for attr_name in BEHAVIOR_CLIP_ATTRIBUTE_KEYS:
                        attr_value = clip_data.attrs[attr_name]
                        # Convert numpy types to python native types
                        if hasattr(attr_value, 'item'):
                            clip_attrs[attr_name] = attr_value.item()
                        else:
                            clip_attrs[attr_name] = attr_value
                    results['clip_attributes'].append(clip_attrs)
                    
                    # Analyze gripper filtering conditions
                    gripper_analysis = analyze_behavior_gripper_filtering(clip_data)
                    gripper_analysis['clip_key'] = clip_key
                    results['gripper_analysis'].append(gripper_analysis)
                    
                    # Check if clip should be skipped
                    should_skip, skip_reason = should_skip_behavior_clip(clip_data)
                    if should_skip:
                        clip_result['status'] = 'skipped'
                        clip_result['issues'].append(f'Skipped due to {skip_reason}')
                        results['clip_details'].append(clip_result)
                        results['skipped_clips'] += 1
                        
                        # Track skip reason
                        if skip_reason not in results['skip_reasons']:
                            results['skip_reasons'][skip_reason] = 0
                        results['skip_reasons'][skip_reason] += 1
                        continue
                
                # Speedup mode - if episode is complete and clip is marked complete, consider it valid
                if use_speedup:
                    clip_complete = clip_data.attrs.get('clip_complete', False)
                    if clip_complete:
                        clip_result['status'] = 'valid'
                        results['clip_details'].append(clip_result)
                        results['valid_clips'] += 1
                        continue
                    else:
                        issue = "Clip not marked as complete despite episode being complete"
                        clip_result['issues'].append(issue)
                        results['clip_details'].append(clip_result)
                        results['invalid_clips'] += 1
                        if debug:
                            raise IntegrityError(f"File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                        else:
                            print(f"INTEGRITY ISSUE - File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                        continue
                
                # Fast mode - only check if clip key contains a colon
                if fastmode:
                    normals_issues = check_clip_normals_contract(
                        clip_data=clip_data,
                        data_keys=data_keys,
                        h5_path=h5_path,
                        clip_key=clip_key,
                    )
                    if normals_issues:
                        issue = f"Normals contract violations: {normals_issues}"
                        clip_result['issues'].append(issue)
                        results['clip_details'].append(clip_result)
                        results['invalid_clips'] += 1
                        if debug:
                            raise IntegrityError(f"File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                        print(f"INTEGRITY ISSUE - File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                        continue
                    clip_result['status'] = 'valid'
                    results['clip_details'].append(clip_result)
                    results['valid_clips'] += 1
                    continue
                
                # Full integrity check mode (original logic)
                if not isinstance(clip_data, (h5py.Group, h5py.Dataset)):
                    issue = "Not a valid data group/dataset"
                    clip_result['issues'].append(issue)
                    results['clip_details'].append(clip_result)
                    results['invalid_clips'] += 1
                    if debug:
                        raise IntegrityError(f"File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                    else:
                        print(f"INTEGRITY ISSUE - File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                    continue
                
                # Check for camera keys
                camera_keys = [k for k in clip_data.keys() if k.startswith('camera_')]
                if not camera_keys:
                    issue = "No valid camera data"
                    clip_result['issues'].append(issue)
                    results['clip_details'].append(clip_result)
                    results['invalid_clips'] += 1
                    if debug:
                        raise IntegrityError(f"File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                    else:
                        print(f"INTEGRITY ISSUE - File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                    continue
                
                # Check scene modalities
                scene_modalities = [m for m in data_keys if 'scene' in m]
                # Camera-specific modalities (image and camera matrix data)
                camera_specific_modalities = [m for m in data_keys if m in IMAGE_KEYS]
                missing_modalities_by_camera = {}
                
                for camera_key in camera_keys:
                    camera_data = clip_data[camera_key]
                    if not isinstance(camera_data, (h5py.Group, h5py.Dataset)):
                        missing_modalities_by_camera[camera_key] = ["invalid camera data format"]
                        continue
                        
                    missing_modalities = []
                    # Check scene modalities
                    for scene_modality in scene_modalities:
                        if not check_exists_and_valid(camera_data, scene_modality, check_write_complete):
                            missing_modalities.append(scene_modality)
                            continue
                        if _is_normals_modality(scene_modality):
                            is_valid, reason = check_normals_payload_contract(
                                camera_data,
                                modality=scene_modality,
                                h5_path=h5_path,
                                clip_key=clip_key,
                                camera_key=camera_key,
                            )
                            if not is_valid:
                                missing_modalities.append(reason)
                    
                    # Check dimension consistency for scene modalities
                    if scene_modalities:
                        is_consistent, expected_shape, mismatched_modalities = check_scene_modality_dimensions(camera_data, scene_modalities, check_write_complete)
                        if not is_consistent:
                            issue_detail = f"Scene modalities dimension mismatch. Expected shape (T, NS): {expected_shape}. Mismatched: {mismatched_modalities}"
                            missing_modalities.append(issue_detail)
                    
                    # Check camera-specific modalities (image and camera matrix data)
                    for camera_modality in camera_specific_modalities:
                        if not check_exists_and_valid(camera_data, camera_modality, check_write_complete):
                            missing_modalities.append(camera_modality)
                            continue
                        is_valid, reason = check_camera_payload_contract(camera_data, camera_modality)
                        if not is_valid:
                            missing_modalities.append(reason)
                            
                    if missing_modalities:
                        missing_modalities_by_camera[camera_key] = missing_modalities
                
                if missing_modalities_by_camera:
                    issue = f"Cameras missing modalities: {missing_modalities_by_camera} (available modalities: {camera_data.keys()})"
                    clip_result['issues'].append(issue)
                    results['clip_details'].append(clip_result)
                    results['invalid_clips'] += 1
                    if debug:
                        raise IntegrityError(f"File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                    else:
                        print(f"INTEGRITY ISSUE - File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                    continue
                
                # Check non-scene modalities (clip-level data, not camera-specific)
                missing_non_scene_modalities = []
                for modality in data_keys:
                    if 'scene' not in modality and modality not in IMAGE_KEYS:
                        # Robot mesh paths are no longer checked here
                        if not check_exists_and_valid(clip_data, modality, check_write_complete):
                            missing_non_scene_modalities.append(modality)
                
                if missing_non_scene_modalities:
                    issue = f"Missing non-scene modalities: {missing_non_scene_modalities}"
                    clip_result['issues'].append(issue)
                    results['clip_details'].append(clip_result)
                    results['invalid_clips'] += 1
                    if debug:
                        raise IntegrityError(f"File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                    else:
                        print(f"INTEGRITY ISSUE - File: {h5_path}, Clip: {clip_key}, Issue: {issue}")
                    continue
                
                # If we get here, all required modalities are present
                clip_result['status'] = 'valid'
                results['clip_details'].append(clip_result)
                results['valid_clips'] += 1
            
            # Update file status
            if results['valid_clips'] == results['total_clips']:
                results['status'] = 'valid'
            elif results['valid_clips'] > 0:
                results['status'] = 'partially_valid'
            else:
                # File has no valid clips - will be skipped
                if not fastmode and results['total_clips'] > 0:
                    print(f"SKIP FILE - {h5_path}: No valid clips (all {results['total_clips']} clips filtered out)")
    
    except IntegrityError:
        # Re-raise in debug mode
        raise
    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
        if debug:
            raise IntegrityError(f"File: {h5_path}, Error: {str(e)}")
        else:
            print(f"INTEGRITY ISSUE - File: {h5_path}, Error: {str(e)}")
    
    return results

def worker_process(file_list: List[str], data_keys: List[str], process_id: int, debug: bool = False, fastmode: bool = False, check_write_complete: bool = False, domain: str = None) -> Dict[str, Any]:
    """Process a list of files and return integrity statistics."""
    validate_domain(domain)
    results = {
        'process_id': process_id,
        'files_checked': 0,
        'valid_files': 0,
        'partially_valid_files': 0,
        'invalid_files': 0,
        'failed_files': 0,
        'total_clips': 0,
        'valid_clips': 0,
        'invalid_clips': 0,
        'skipped_clips': 0,
        'skip_reasons': {},  # Track aggregate skip reasons
        'file_results': [],
        'all_clip_attributes': [],  # For Behavior domain attribute collection
        'all_gripper_analysis': []  # For Behavior domain gripper filtering analysis
    }
    
    disable_tqdm = os.environ.get("PW_DISABLE_TQDM", "0") == "1"
    for h5_path in tqdm(
        file_list,
        desc=f"Worker {process_id}",
        position=process_id,
        disable=disable_tqdm,
    ):
        file_result = check_file_integrity(h5_path, data_keys, debug, fastmode, check_write_complete, domain)
        results['files_checked'] += 1
        results['total_clips'] += file_result['total_clips']
        results['valid_clips'] += file_result['valid_clips']
        results['invalid_clips'] += file_result['invalid_clips']
        results['skipped_clips'] += file_result.get('skipped_clips', 0)
        
        # Collect clip attributes for Behavior domain
        if domain == 'behavior' and 'clip_attributes' in file_result:
            results['all_clip_attributes'].extend(file_result['clip_attributes'])
        
        # Collect gripper analysis for Behavior domain
        if domain == 'behavior' and 'gripper_analysis' in file_result:
            results['all_gripper_analysis'].extend(file_result['gripper_analysis'])
        
        # Aggregate skip reasons
        if 'skip_reasons' in file_result:
            for reason, count in file_result['skip_reasons'].items():
                if reason not in results['skip_reasons']:
                    results['skip_reasons'][reason] = 0
                results['skip_reasons'][reason] += count
        
        if file_result['status'] == 'valid':
            results['valid_files'] += 1
        elif file_result['status'] == 'partially_valid':
            results['partially_valid_files'] += 1
        elif file_result['status'] == 'failed':
            results['failed_files'] += 1
        else:
            results['invalid_files'] += 1
            
        results['file_results'].append(file_result)
        
    return results

def aggregate_results(worker_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from all workers."""
    aggregate = {
        'files_checked': 0,
        'valid_files': 0,
        'partially_valid_files': 0,
        'invalid_files': 0,
        'failed_files': 0,
        'total_clips': 0,
        'valid_clips': 0,
        'invalid_clips': 0,
        'skipped_clips': 0,
        'skip_reasons': {},  # Track aggregate skip reasons
        'issue_counts': Counter(),
        'file_results': [],
        'all_clip_attributes': [],  # For Behavior domain attribute collection
        'all_gripper_analysis': []  # For Behavior domain gripper filtering analysis
    }
    
    for result in worker_results:
        aggregate['files_checked'] += result['files_checked']
        aggregate['valid_files'] += result['valid_files']
        aggregate['partially_valid_files'] += result['partially_valid_files']
        aggregate['invalid_files'] += result['invalid_files']
        aggregate['failed_files'] += result['failed_files']
        aggregate['total_clips'] += result['total_clips']
        aggregate['valid_clips'] += result['valid_clips']
        aggregate['invalid_clips'] += result['invalid_clips']
        aggregate['skipped_clips'] += result.get('skipped_clips', 0)
        
        # Aggregate clip attributes for Behavior domain
        if 'all_clip_attributes' in result:
            aggregate['all_clip_attributes'].extend(result['all_clip_attributes'])
        
        # Aggregate gripper analysis for Behavior domain
        if 'all_gripper_analysis' in result:
            aggregate['all_gripper_analysis'].extend(result['all_gripper_analysis'])
        
        # Aggregate skip reasons
        if 'skip_reasons' in result:
            for reason, count in result['skip_reasons'].items():
                if reason not in aggregate['skip_reasons']:
                    aggregate['skip_reasons'][reason] = 0
                aggregate['skip_reasons'][reason] += count
        
        for file_result in result['file_results']:
            aggregate['file_results'].append(file_result)
            
            # Count issues for statistics
            for clip_detail in file_result.get('clip_details', []):
                if clip_detail['status'] == 'invalid':
                    for issue in clip_detail.get('issues', []):
                        if isinstance(issue, str):
                            aggregate['issue_counts'][issue] += 1
                        elif isinstance(issue, dict):
                            # For camera missing modalities
                            aggregate['issue_counts']['Camera missing modalities'] += 1
    
    return aggregate

def generate_report(results: Dict[str, Any], domain: str = None) -> str:
    """Generate a formatted report of the integrity check results."""
    report_lines = []
    report_lines.append('\n'*50)
    report_lines.append("\n" + "="*80)
    report_lines.append(" "*30 + "DATASET INTEGRITY REPORT")
    report_lines.append("="*80 + "\n")
    
    report_lines.append(f"Total files checked: {results['files_checked']}")
    report_lines.append(f"Files with all valid clips: {results['valid_files']} ({results['valid_files']/results['files_checked']*100:.2f}%)")
    report_lines.append(f"Files with some valid clips: {results['partially_valid_files']} ({results['partially_valid_files']/results['files_checked']*100:.2f}%)")
    report_lines.append(f"Files with no valid clips: {results['invalid_files']} ({results['invalid_files']/results['files_checked']*100:.2f}%)")
    report_lines.append(f"Failed to process files: {results['failed_files']} ({results['failed_files']/results['files_checked']*100:.2f}%)")
    
    report_lines.append("\nCLIP STATISTICS:")
    report_lines.append(f"Total clips checked: {results['total_clips']}")
    if results['total_clips'] > 0:
        report_lines.append(f"Valid clips: {results['valid_clips']} ({results['valid_clips']/results['total_clips']*100:.2f}%)")
        report_lines.append(f"Invalid clips: {results['invalid_clips']} ({results['invalid_clips']/results['total_clips']*100:.2f}%)")
        if 'skipped_clips' in results and results['skipped_clips'] > 0:
            report_lines.append(f"Skipped clips: {results['skipped_clips']} ({results['skipped_clips']/results['total_clips']*100:.2f}%)")
    else:
        report_lines.append(f"Valid clips: {results['valid_clips']} (N/A%)")
        report_lines.append(f"Invalid clips: {results['invalid_clips']} (N/A%)")
        if 'skipped_clips' in results:
            report_lines.append(f"Skipped clips: {results['skipped_clips']} (N/A%)")
    
    # Skip reason breakdown for Behavior domain
    if domain == 'behavior' and 'skip_reasons' in results and results['skip_reasons']:
        report_lines.append("\nSKIP REASON BREAKDOWN:")
        total_skipped = results.get('skipped_clips', 0)
        if total_skipped > 0:
            for reason, count in sorted(results['skip_reasons'].items()):
                percentage = (count / total_skipped) * 100
                report_lines.append(f"- {reason}: {count} clips ({percentage:.2f}% of skipped clips)")
        else:
            report_lines.append("No clips were skipped")
    
    report_lines.append("\nTOP ISSUES:")
    for issue, count in results['issue_counts'].most_common(10):
        report_lines.append(f"- {issue}: {count} occurrences")
    
    # Behavior domain attribute statistics
    if domain == 'behavior' and 'all_clip_attributes' in results and results['all_clip_attributes']:
        report_lines.append("\n" + "="*80)
        report_lines.append(" "*25 + "Behavior CLIP ATTRIBUTE STATISTICS")
        report_lines.append("="*80)
        
        attr_stats = calculate_behavior_attribute_stats(results['all_clip_attributes'])
        if attr_stats:
            report_lines.append(f"{'Attribute':<35} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12} {'Count':<8}")
            report_lines.append("-" * 80)
            for attr_name, stats in sorted(attr_stats.items()):
                report_lines.append(f"{attr_name:<35} {stats['min']:<12.6f} {stats['max']:<12.6f} "
                      f"{stats['mean']:<12.6f} {stats['std']:<12.6f} {stats['count']:<8}")
        else:
            report_lines.append("No numeric attributes found in clip data.")
    
    # Behavior domain gripper filtering statistics
    if domain == 'behavior' and 'all_gripper_analysis' in results and results['all_gripper_analysis']:
        report_lines.append("\n" + "="*80)
        report_lines.append(" "*20 + "Behavior GRIPPER FILTERING ANALYSIS")
        report_lines.append("="*80)
        
        gripper_stats = calculate_behavior_gripper_filtering_stats(results['all_gripper_analysis'])
        if 'error' in gripper_stats:
            report_lines.append(f"Error: {gripper_stats['error']}")
        else:
            total_clips = gripper_stats['total_clips_analyzed']
            report_lines.append(f"Total clips analyzed: {total_clips}")
            report_lines.append("")
            
            # Individual conditions
            report_lines.append("INDIVIDUAL CONDITIONS:")
            report_lines.append(f"{'Condition':<35} {'Count':<10} {'Percentage':<12}")
            report_lines.append("-" * 60)
            for condition, data in gripper_stats['individual_conditions'].items():
                condition_name = condition.replace('_', ' ').title()
                report_lines.append(f"{condition_name:<35} {data['count']:<10} {data['percentage']:<12.2f}%")
            report_lines.append("")
            
            # Combined conditions (eligible for skipping)
            report_lines.append("GRIPPER SKIP ELIGIBILITY (No Collision AND Distance > 0.2m):")
            report_lines.append(f"{'Gripper':<35} {'Count':<10} {'Percentage':<12}")
            report_lines.append("-" * 60)
            combined = gripper_stats['combined_conditions']
            report_lines.append(f"{'Left Gripper Skip Eligible':<35} {combined['left_gripper_skip_eligible']['count']:<10} {combined['left_gripper_skip_eligible']['percentage']:<12.2f}%")
            report_lines.append(f"{'Right Gripper Skip Eligible':<35} {combined['right_gripper_skip_eligible']['count']:<10} {combined['right_gripper_skip_eligible']['percentage']:<12.2f}%")
            report_lines.append(f"{'Both Grippers Skip Eligible':<35} {combined['both_grippers_skip_eligible']['count']:<10} {combined['both_grippers_skip_eligible']['percentage']:<12.2f}%")
            report_lines.append("")
            
            # Distance statistics
            if 'distance_statistics' in gripper_stats:
                report_lines.append("DISTANCE STATISTICS:")
                dist_stats = gripper_stats['distance_statistics']
                
                if 'left_gripper_distances' in dist_stats:
                    left_stats = dist_stats['left_gripper_distances']
                    report_lines.append(f"Left Gripper Min Distance to Objects:")
                    report_lines.append(f"  Min: {left_stats['min']:.4f}m, Max: {left_stats['max']:.4f}m")
                    report_lines.append(f"  Mean: {left_stats['mean']:.4f}m, Std: {left_stats['std']:.4f}m")
                    report_lines.append(f"  Above 0.2m threshold: {left_stats['threshold_0_2']} clips ({left_stats['threshold_0_2_percentage']:.2f}%)")
                    report_lines.append("")
                
                if 'right_gripper_distances' in dist_stats:
                    right_stats = dist_stats['right_gripper_distances']
                    report_lines.append(f"Right Gripper Min Distance to Objects:")
                    report_lines.append(f"  Min: {right_stats['min']:.4f}m, Max: {right_stats['max']:.4f}m")
                    report_lines.append(f"  Mean: {right_stats['mean']:.4f}m, Std: {right_stats['std']:.4f}m")
                    report_lines.append(f"  Above 0.2m threshold: {right_stats['threshold_0_2']} clips ({right_stats['threshold_0_2_percentage']:.2f}%)")
    
    report_lines.append("\n" + "="*80)
    report_lines.append(" "*30 + "END OF REPORT")
    report_lines.append("="*80 + "\n")
    
    return "\n".join(report_lines)

def print_report(results: Dict[str, Any], domain: str = None):
    """Print a formatted report of the integrity check results."""
    report_text = generate_report(results, domain)
    print(report_text)

def save_report_to_file(results: Dict[str, Any], domain: str = None, filepath: str = None):
    """Save the formatted report to a text file."""
    if filepath is None:
        raise ValueError("filepath must be provided")
    
    report_text = generate_report(results, domain)
    
    with open(filepath, 'w') as f:
        f.write(report_text)
    
    print(f"Integrity report saved to {filepath}")

def extract_valid_clips(results: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Extract valid (h5_path, clip_key) pairs from aggregated results.
    
    Args:
        results: Aggregated results from integrity check
        
    Returns:
        List of valid (h5_path, clip_key) pairs
    """
    valid_clips = []
    
    for file_result in results['file_results']:
        h5_path = file_result['file_path']
        
        for clip_detail in file_result.get('clip_details', []):
            if clip_detail['status'] == 'valid':
                clip_key = clip_detail['clip_key']
                clip_pair = (h5_path, clip_key)
                valid_clips.append(clip_pair)
    
    return valid_clips

def prepare_integrity_result(aggregate_result: Dict[str, Any], data_keys: List[str], fastmode: bool = False) -> Dict[str, Any]:
    """
    Prepare a simplified integrity result for saving/loading.
    
    Args:
        aggregate_result: Aggregated results from integrity check
        data_keys: List of data modalities checked
        fastmode: Whether fast mode was used (only checking clip key format)
        
    Returns:
        Dict with simplified results including valid clip pairs
    """
    valid_clips = extract_valid_clips(aggregate_result)
    
    result = {
        'valid_clips': valid_clips,
        'stats': {
            'total_files': aggregate_result['files_checked'],
            'valid_files': aggregate_result['valid_files'],
            'total_clips': aggregate_result['total_clips'],
            'valid_clips': aggregate_result['valid_clips'],
            'timestamp': time.time()
        },
        'config': {
            'data_keys': data_keys,
            'fastmode': fastmode,
        }
    }
    
    # Add Behavior domain specific data if available
    if 'all_clip_attributes' in aggregate_result:
        result['behavior_clip_attributes'] = aggregate_result['all_clip_attributes']
    
    if 'all_gripper_analysis' in aggregate_result:
        result['behavior_gripper_analysis'] = aggregate_result['all_gripper_analysis']
        # Also include computed statistics
        result['behavior_gripper_statistics'] = calculate_behavior_gripper_filtering_stats(aggregate_result['all_gripper_analysis'])
    
    return result

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if hasattr(obj, 'item'):
        # For numpy scalars
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def save_integrity_results(results: Dict[str, Any], filepath: str):
    """Save integrity check results to a JSON file."""
    # Convert any numpy types to Python native types
    json_safe_results = convert_numpy_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(json_safe_results, f)
    print(f"Integrity results saved to {filepath}")

def save_behavior_detailed_analysis(results: Dict[str, Any], output_dir: str):
    """
    Save detailed Behavior domain analysis to separate files.
    
    Args:
        results: Aggregated results containing Behavior analysis data
        output_dir: Directory to save the detailed analysis files
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save gripper filtering statistics
    if 'behavior_gripper_statistics' in results:
        gripper_stats_file = os.path.join(output_dir, 'behavior_gripper_filtering_statistics.json')
        with open(gripper_stats_file, 'w') as f:
            json.dump(convert_numpy_types(results['behavior_gripper_statistics']), f, indent=2)
        print(f"Behavior gripper filtering statistics saved to {gripper_stats_file}")
    
    # Save raw gripper analysis data (optional, for debugging)
    if 'behavior_gripper_analysis' in results:
        gripper_analysis_file = os.path.join(output_dir, 'behavior_gripper_analysis_raw.json')
        with open(gripper_analysis_file, 'w') as f:
            json.dump(convert_numpy_types(results['behavior_gripper_analysis']), f, indent=2)
        print(f"Behavior raw gripper analysis data saved to {gripper_analysis_file}")
    
    # Save clip attributes
    if 'behavior_clip_attributes' in results:
        clip_attrs_file = os.path.join(output_dir, 'behavior_clip_attributes.json')
        with open(clip_attrs_file, 'w') as f:
            json.dump(convert_numpy_types(results['behavior_clip_attributes']), f, indent=2)
        print(f"Behavior clip attributes saved to {clip_attrs_file}")
        
        # Also save attribute statistics
        attr_stats = calculate_behavior_attribute_stats(results['behavior_clip_attributes'])
        if attr_stats:
            attr_stats_file = os.path.join(output_dir, 'behavior_attribute_statistics.json')
            with open(attr_stats_file, 'w') as f:
                json.dump(convert_numpy_types(attr_stats), f, indent=2)
            print(f"Behavior attribute statistics saved to {attr_stats_file}")

def run_integrity_check(
    input_dir: str,
    data_keys: List[str],
    num_mp_workers: int = None,
    output_file: str = None,
    debug: bool = False,
    fastmode: bool = False,
    check_write_complete: bool = False,
    domain: str = None
) -> Dict[str, Any]:
    """
    Run integrity check on the dataset and return valid clips.
    
    Args:
        input_dir: Directory containing .h5 files
        data_keys: List of required data modalities
        num_mp_workers: Number of multiprocessing workers (defaults to CPU count)
        output_file: Optional path to save results to
        debug: If True, uses a single worker and raises the first issue encountered
        fastmode: If True, only checks clip key format without validating contents
        check_write_complete: Whether to check for write_complete attribute on datasets
        
    Returns:
        Dict containing valid clips and statistics
    """
    start_time = time.time()
    validate_domain(domain)
    
    # Force single worker in debug mode
    if debug:
        print("Running in DEBUG mode: Using 1 worker and will stop at first issue")
        num_mp_workers = 1
    elif num_mp_workers is None:
        num_mp_workers = mp.cpu_count()
    
    # Gather all files
    h5_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".h5") or file.endswith(".hdf5"):
                full_path = os.path.join(root, file)
                h5_files.append(full_path)
    
    if len(h5_files) == 0:
        raise ValueError(f"No .h5/.hdf5 files found in {input_dir}")
    
    print(f"Found {len(h5_files)} H5 files in {input_dir}")
    print(f"Using {num_mp_workers} workers to check integrity")
    if fastmode:
        print("Running in FASTMODE: Only checking clip key format without validating contents")
    if check_write_complete:
        print("Checking for write_complete attribute on all datasets")
    
    # Handle special case for single worker - no need for multiprocessing
    if num_mp_workers == 1:
        worker_results = [worker_process(
            file_list=h5_files,
            data_keys=data_keys,
            process_id=0,
            debug=debug,
            fastmode=fastmode,
            check_write_complete=check_write_complete,
            domain=domain
        )]
    else:
        # Split files among workers
        files_per_worker = [[] for _ in range(num_mp_workers)]
        for i, file_path in enumerate(h5_files):
            files_per_worker[i % num_mp_workers].append(file_path)
        
        # Start worker processes
        pool = mp.Pool(processes=num_mp_workers)
        
        worker_results = pool.starmap(
            worker_process, 
            [(files, data_keys, i, False, fastmode, check_write_complete, domain) 
             for i, files in enumerate(files_per_worker)]
        )
        
        pool.close()
        pool.join()
    
    # Aggregate results
    aggregate_result = aggregate_results(worker_results)
    
    # Prepare simplified result
    integrity_result = prepare_integrity_result(
        aggregate_result, data_keys, fastmode
    )
    
    # Save results if output file provided
    if output_file is None:
        output_file = os.path.join(input_dir, "integrity_check.json")
    save_integrity_results(integrity_result, output_file)
    
    # Save Behavior detailed analysis to separate files if Behavior domain
    if domain == 'behavior' and ('behavior_gripper_analysis' in integrity_result or 'behavior_clip_attributes' in integrity_result):
        # Create output directory for detailed analysis
        output_dir = os.path.dirname(output_file)
        detailed_analysis_dir = os.path.join(output_dir, "behavior_detailed_analysis")
        save_behavior_detailed_analysis(integrity_result, detailed_analysis_dir)
    
    # Save complete console report to file
    output_dir = os.path.dirname(output_file)
    report_filename = os.path.splitext(os.path.basename(output_file))[0] + "_report.txt"
    report_filepath = os.path.join(output_dir, report_filename)
    save_report_to_file(aggregate_result, domain, report_filepath)
    
    # Print report
    print_report(aggregate_result, domain)
    
    elapsed_time = time.time() - start_time
    print(f"Integrity check completed in {elapsed_time:.2f} seconds")
    
    return integrity_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run integrity checks on H5 outputs.")
    parser.add_argument("--input_dir", required=True, help="Directory containing .h5/.hdf5 files.")
    parser.add_argument(
        "--domain",
        required=True,
        choices=["droid", "behavior"],
        help="Dataset domain to validate.",
    )
    parser.add_argument("--num_mp_workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save integrity JSON.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (single worker, fail fast).")
    parser.add_argument("--fastmode", action="store_true", help="Skip heavy validation and only check clip keys.")
    parser.add_argument("--check_write_complete", action="store_true", help="Require write_complete on datasets.")
    args = parser.parse_args()
    validate_domain(args.domain)

    data_keys = get_wds_data_keys(args.domain)

    run_integrity_check(
        input_dir=args.input_dir,
        data_keys=data_keys,
        num_mp_workers=args.num_mp_workers,
        output_file=args.output_file,
        debug=args.debug,
        fastmode=args.fastmode,
        check_write_complete=args.check_write_complete,
        domain=args.domain,
    )


if __name__ == "__main__":
    main()
