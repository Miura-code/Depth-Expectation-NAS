

from config.searchStage_HINT_config import SearchStageHintConfig
from config.searchStage_config import SearchStageConfig
import searchStage_ArchHint_main
import searchStage_Hint_main
import searchStage_KD_main
import searchevaluateStage_main

def main():
    config = SearchStageConfig()
    if "KD" in config.type:
        searchStage_KD_main.run_task(config)
    elif config.type == "ArchHINT":
        searchStage_ArchHint_main.run_task(config)
    elif config.type == "SearchEval":
        searchevaluateStage_main.run_task(config)
    else:
        raise NotImplementedError("実装されていない学習手法です")


if __name__ == "__main__":
    main()
