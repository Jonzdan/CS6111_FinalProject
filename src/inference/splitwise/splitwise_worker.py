from shared.base_worker import BaseWorker
from vllm.config.kv_transfer import KVTransferConfig
from vllm import LLM

class SplitWiseWorker(BaseWorker):
    def __init__(self, kv_transfer_config: KVTransferConfig):
        self.llm = LLM(
            model='llama_eagle_3',
            kv_transfer_config=kv_transfer_config
        )
    
    def inference(self, prompts):
        return self.llm.generate(prompts)
    

# To make it more complicated, add worker node prediction (fair KV cache allocation -> migration)
# Most likely reimplemnet some part of the KV cache transfer