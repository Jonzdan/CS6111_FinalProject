import vllm

class BaseWorker:
    def __init__(self, **kwargs):
        self.llm = vllm.LLM(model='llama_eagle_3.py', **kwargs)

    def inference(self, prompts: list[str]):
        outputs = self.llm.generate(
            prompts=prompts
        )

        return outputs
    


