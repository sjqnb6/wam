# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import logging
from typing import Any

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.import_utils import require_package


class SwanLabLogger:
    """Log LeRobot training metrics to SwanLab."""

    def __init__(self, cfg: TrainPipelineConfig) -> None:
        require_package("swanlab", extra="swanlab")
        import swanlab

        self._swanlab = swanlab
        init_kwargs: dict[str, Any] = {
            "project": cfg.swanlab.project,
            "workspace": cfg.swanlab.workspace,
            "experiment_name": cfg.job_name,
            "config": cfg.to_dict(),
            "mode": cfg.swanlab.mode,
        }
        if cfg.swanlab.run_id is not None:
            # A run ID is saved in each training checkpoint's train_config.json.
            # Reuse it when resuming so metrics continue in the same SwanLab run.
            init_kwargs["id"] = cfg.swanlab.run_id
            init_kwargs["resume"] = "allow"

        run = self._swanlab.init(**init_kwargs)
        if cfg.swanlab.run_id is None:
            cfg.swanlab.run_id = run.id

        logging.info(
            "SwanLab logging enabled for project %s (run_id=%s)", cfg.swanlab.project, cfg.swanlab.run_id
        )

    def log_dict(self, values: dict[str, Any], step: int, mode: str = "train") -> None:
        """Log scalar values with the same train/eval namespaces used by WandB."""
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        data = {
            f"{mode}/{key}": value
            for key, value in values.items()
            if isinstance(value, (int, float))
        }
        if data:
            self._swanlab.log(data, step=step)

    def finish(self) -> None:
        self._swanlab.finish()
