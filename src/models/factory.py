def build_model(model_cfg):
    name = (model_cfg.get("name") or "").upper()

    if name == "MAE":
        from src.models.MAE import build_mae_model

        return build_mae_model(model_cfg)
    if name == "MAE-IN":
        from src.models.MAE_IN import build_mae_in_model

        return build_mae_in_model(model_cfg)
    if name == "CLASSIFIER_MAE":
        from src.models.classifier import build_classifier

        return build_classifier(model_cfg)

    raise ValueError(f"Unsupported model name: {name}")
