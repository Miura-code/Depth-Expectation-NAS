# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# Modifications made by Shun Miura(https://github.com/Miura-code)

from config.searchStage_config import SearchStageConfig
import searchevaluateStage_main
import trainer_runner

def main():
    config = SearchStageConfig()
    if "SearchEval" in config.type:
        searchevaluateStage_main.run_task(config)
    else:
        trainer_runner.run_task(config)

if __name__ == "__main__":
    main()
