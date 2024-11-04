

from config.searchStage_config import SearchStageConfig
import searchStage_ArchHint_main
import searchevaluateStage_main
import trainer_runner

def main():
    config = SearchStageConfig()
    if config.type == "ArchHINT":
        searchStage_ArchHint_main.run_task(config)
    elif config.type == "SearchEval":
        searchevaluateStage_main.run_task(config)
    else:
        trainer_runner.run_task(config)

if __name__ == "__main__":
    main()
