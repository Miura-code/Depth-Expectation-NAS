# Contact: https://github.com/Miura-code

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
