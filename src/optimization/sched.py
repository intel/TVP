# Copyright (C) 2022 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0

"""
optimizer learning rate scheduling helpers
"""
from math import ceil
from collections import Counter


def noam_schedule(step, warmup_step=4000):
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))


def multi_step_schedule(n_epoch, milestones, gamma=0.5):
    milestones = list(sorted(milestones))
    for i, m in enumerate(milestones):
        if n_epoch < m:
            return gamma**i
    return gamma**(len(milestones)+1)


def get_lr_sched(global_step, decay, learning_rate,
                 num_train_steps, warmup_ratio=0.1,
                 decay_epochs=[], multi_step_epoch=-1):
    warmup_steps = int(warmup_ratio*num_train_steps)
    if decay == 'linear':
        lr_this_step = learning_rate * warmup_linear(
            global_step, warmup_steps, num_train_steps)
    elif decay == 'invsqrt':
        lr_this_step = learning_rate * noam_schedule(
            global_step, warmup_steps)
    elif decay == 'constant':
        lr_this_step = learning_rate
    elif decay == "multi_step":
        assert multi_step_epoch >= 0
        lr_this_step = learning_rate * multi_step_schedule(
            multi_step_epoch, decay_epochs)
    if lr_this_step <= 0:
        # save guard for possible miscalculation of train steps
        lr_this_step = 1e-8
    return lr_this_step
