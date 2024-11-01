class SAM2:
    def __init__(self, model_name,
                 hf_token=None):
        pass


class SAM2Serve:
    loaded_models = {}

    def get_or_load_model(self, model_name, **kwargs):
        model = self.loaded_models.get(model_name)
        if not model:
            self.loaded_models[model_name] = SAM2(model_name, **kwargs)
            return self.loaded_models[model_name]
        else:
            return model
