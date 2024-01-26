class Llama2TokenizeTest():
    def __init__(self) -> None:
        import transformers
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ")
    
    def show_tokens(self, text):
        for token_id in self.tokenizer.encode(text):
            print(token_id, self.tokenizer.decode([token_id]))

    def interactive(self):
        while True:
            text = input("Enter text: ")
            self.show_tokens(text)