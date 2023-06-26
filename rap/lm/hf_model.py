from .. import LanguageModel

class DummyLM(LanguageModel):
    def query(self, query: str) -> float:
        return query + "|"
    