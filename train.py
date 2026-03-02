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

import sys
import traceback
import torch.distributed as dist
import wandb
from arguments import parse_args
from training.trainer import Trainer
from utils import _print, NaNDetectionError


if __name__ == "__main__":
    trainer = None
    args = None
    training_status = 1
    try:
        args = parse_args()
        trainer = Trainer(args)
        training_status = trainer.train()
    except KeyboardInterrupt:
        training_status = 130
        if trainer and trainer.rank == 0:
            _print("Training interrupted by user.")
    except NaNDetectionError as nan_error:
        training_status = 2
        if trainer and trainer.rank == 0:
            _print(f"Training stopped due to NaN detection: {nan_error}")
    except Exception:
        _print("Training failed with exception:")
        traceback.print_exc()
        training_status = 1
    finally:
        if trainer and trainer.rank == 0:
            wandb.finish(exit_code=training_status)
        if args and args.distributed:
            dist.barrier()
        sys.exit(training_status)
