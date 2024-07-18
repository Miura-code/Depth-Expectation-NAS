def set_params_lr(model, lr):
    """ モデルの特徴抽出層と分類層に別々のLearning Rateを設定する """
    classifier = model.get_classifier()
    features = model.get_features()
    params=[
            {"params": features.parameters(), "lr": lr[0]},
            # {"params": model.features[4:].parameters(), "lr": 0.01},
            {"params": classifier.parameters(), "lr": lr[1]},
    ]
    return params