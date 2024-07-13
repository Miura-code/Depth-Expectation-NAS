def Exception_pretrained_model(model_name: str, cifar: bool = False, pretrained: bool = False):
    """ モデルを正しくロードできるかチェックして例外を発生させる
    Args:
        model_name: モデル名
        cifar: CIFARデータセットを使うかどうか
        pretrained: 事前学習済み重みを使うかどうか
    """
    exeption_log = ""
    if "densenet_cifar" in model_name:
        if pretrained:
            exeption_log = "モデル '{}' には事前学習済み重みはありません。"\
                "通常モデル（DenseNet121等）をロードしてCIFAR用に最終層の付け替えを行ってください".format(model_name)
            raise KeyError(exeption_log)
    else:
        pass