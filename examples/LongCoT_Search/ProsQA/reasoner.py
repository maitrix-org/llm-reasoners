class ProsQAReasoner():
    def __init__(self, base_model, n_sc=1, temperature=0, bs=1):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        
    def __call__(self, example, prompt=None):
        outputs = self.base_model.generate([example], temeprature = self.temperature, hide_input=True).text[0]
        return outputs