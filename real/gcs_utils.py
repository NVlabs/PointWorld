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
"""Google Cloud Storage helpers for DROID data paths."""

import os
import re
import tempfile
import subprocess

def is_gcs_path(path):
    """Check if the path is a Google Cloud Storage path."""
    return path.startswith('gs://')


def parse_gcs_path(gcs_path):
    """Parse a Google Cloud Storage path into bucket and blob names."""
    match = re.match(r'gs://([^/]+)/(.*)', gcs_path)
    if not match:
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    return match.group(1), match.group(2)


def download_from_gcs(gcs_path, local_path):
    """Download a file from Google Cloud Storage to a local path using gsutil."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Use gsutil to download the file
    cmd = ["gsutil", "cp", gcs_path, local_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"GCS download failed for {gcs_path}: {e.stderr.strip()}"
        ) from e
    
    return local_path


def get_local_path(path, temp_dir=None):
    """
    If the path is a GCS path, download the file to a temporary directory and return the local path.
    Otherwise, return the original path.
    
    If POINTWORLD_CACHE_DIR environment variable is set, use it as a persistent
    cache directory (under <POINTWORLD_CACHE_DIR>/droid) instead of temporary
    directories for GCS files.
    """
    if not is_gcs_path(path):
        return path
    
    # Check if caching is enabled
    cache_root = os.environ.get('POINTWORLD_CACHE_DIR')
    
    if cache_root:
        # Use cache directory - ignore temp_dir when caching is enabled
        # Recreate the GCS path structure in the cache directory
        bucket, blob = parse_gcs_path(path)
        local_path = os.path.join(cache_root, "droid", bucket, blob)
        
        # Create directory structure if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Check if file already exists in cache
        if os.path.exists(local_path):
            return local_path
        
        # File not in cache, download it
        download_from_gcs(path, local_path)
        return local_path
    else:
        # Original behavior - use temporary directory
        # Create a temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        # Create local path
        local_path = os.path.join(temp_dir, os.path.basename(path))
        
        # Download the file
        download_from_gcs(path, local_path)
        
        return local_path


def enforce_gcs_cache_policy(
    scene_paths,
    stage_name,
    require_cache=False,
    allow_streaming=False,
):
    """
    Validate cache policy for GCS-backed scene paths.

    Args:
        scene_paths: Iterable of scene paths (local or gs://).
        stage_name: Name of the calling stage (for logs/errors).
        require_cache: If True, fail when GCS input is used without POINTWORLD_CACHE_DIR.
        allow_streaming: If True, bypass require_cache and continue with warning.

    Returns:
        bool: True if at least one GCS path is detected, else False.
    """
    if isinstance(scene_paths, str):
        scene_paths = [scene_paths]

    gcs_paths = [p for p in scene_paths if is_gcs_path(p)]
    if not gcs_paths:
        return False

    cache_root = os.environ.get("POINTWORLD_CACHE_DIR", "").strip()
    if cache_root:
        print(f"[{stage_name}] cache enabled: POINTWORLD_CACHE_DIR={cache_root}")
        return True

    message = (
        f"[{stage_name}] Detected {len(gcs_paths)} GCS scene path(s) but POINTWORLD_CACHE_DIR is not set. "
        "Without persistent caching, repeated stages may re-download the same objects and increase GCS traffic."
    )
    setup_hint = (
        "Set a shared cache root before running this stage, for example: "
        "export POINTWORLD_CACHE_DIR=/path/to/fast_disk/pointworld_cache"
    )

    if require_cache and not allow_streaming:
        raise RuntimeError(
            f"{message} {setup_hint} "
            "If you intentionally want to stream again, rerun with --allow_gcs_streaming."
        )

    if allow_streaming:
        print(
            "WARNING: "
            f"{message} Continuing because --allow_gcs_streaming was provided."
        )
    else:
        print(f"WARNING: {message} {setup_hint}")

    return True

def list_gcs_files(gcs_path, pattern=None):
    """
    List files in a Google Cloud Storage directory that match a pattern using gsutil.
    
    Args:
        gcs_path (str): GCS path to the directory
        pattern (str, optional): Glob pattern to filter files
        
    Returns:
        list: List of matching GCS paths
    """
    # Make sure path ends with a slash if it's a directory and doesn't already have one
    if not gcs_path.endswith('/'):
        gcs_path += '/'
    
    # Construct the gsutil command
    if pattern:
        # Add the pattern to the path
        search_path = os.path.join(gcs_path, pattern)
    else:
        search_path = gcs_path + '*'
    
    cmd = ["gsutil", "ls", search_path]
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Split the output into lines and remove empty lines
        paths = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        return paths
    except subprocess.CalledProcessError as e:
        if "No URLs matched" in e.stderr:
            # No files matched the pattern
            return []
        else:
            raise RuntimeError(f"Error listing files in {gcs_path}: {e.stderr}")
