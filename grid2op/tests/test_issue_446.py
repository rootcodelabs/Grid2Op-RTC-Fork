# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import grid2op
from grid2op.gym_compat import BoxGymObsSpace
import numpy as np
import unittest
import warnings


class Issue446Tester(unittest.TestCase):
    def test_box_action_space(self):
        # We considers only redispatching actions
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)

        divide = {"hour_of_day": np.ones(1)}
        subtract = {"hour_of_day": np.zeros(1)}

        gym_observation_space_1 =  BoxGymObsSpace(env.observation_space,
                                                attr_to_keep=["curtailment_mw", "hour_of_day"],
                                                divide = divide,
                                                subtract = subtract
                                                )

        gym_observation_space_2 =  BoxGymObsSpace(env.observation_space.copy(),
                                                attr_to_keep=["curtailment_mw", "hour_of_day"],
                                                divide = divide,
                                                subtract = subtract
                                                )
        
        gym_observation_space_1.normalize_attr("curtailment_mw")
            
        assert "curtailment_mw" in gym_observation_space_1._divide
        assert "curtailment_mw" not in gym_observation_space_2._divide
        assert "curtailment_mw" in gym_observation_space_1._subtract
        assert "curtailment_mw" not in gym_observation_space_2._subtract
        
        
if __name__ == "__main__":
    unittest.main()