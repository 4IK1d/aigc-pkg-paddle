"""Prepare data for NLG tasks, such as title generation, abstract generation.
"""

import abc

# import paddle
# import paddle.nn as nn
from paddle.io import BatchSampler, DistributedBatchSampler, DataLoader
# from paddlenlp.transformers import PegasusForConditionalGeneration, PegasusChineseTokenizer
# from paddlenlp.transformers import LinearDecayWithWarmup
# from paddlenlp.utils.log import logger
# from paddlenlp.metrics import BLEU
from paddlenlp.data import DataCollatorForSeq2Seq

class ModelSet:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

class AbstractPrepareConfig:
    def __init__(self, text_column, target_column, max_source_length, max_target_length, min_target_length, batch_size):
        """
        # 原始字段需要移除
        remove_columns = ['content', 'title']
        # 文本的最大长度
        max_source_length = 128
        # 摘要的最大长度
        max_target_length = 64
        """
        self.text_column = text_column 
        self.target_column = target_column 
        self.max_source_length = max_source_length 
        self.max_target_length = max_target_length
        self.min_target_length = min_target_length 
        self.batch_size = batch_size
 
class BasePrepare(abc.ABC):
    """Base class for preparation.
    """
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def get_dataset(self):
        pass

    @abc.abstractmethod
    def get_dataloader(self):
        pass

class AbstractGenerationPrapare(BasePrepare):
    """Prepare data for NLG tasks abstract generation.
    """
    def __init__(self, model_config: ModelSet, prepare_config: AbstractPrepareConfig):
        super(AbstractGenerationPrapare, self).__init__()
        self.model_config = model_config
        self.prepare_config = prepare_config

    def _convert_example(self, example):
        """
        构造模型的输入.
        """
        inputs = example[self.prepare_config.text_column]
        targets = example[self.prepare_config.target_column]
        # 分词
        model_inputs = self.model_config.tokenizer(inputs,
                                max_length=self.prepare_config.max_source_length,
                                padding=False,
                                truncation=True,
                                return_attention_mask=True)
        labels = self.model_config.tokenizer(targets,
                        max_length=self.prepare_config.max_target_length,
                        padding=False,
                        truncation=True)
        # 得到labels，后续通过DataCollatorForSeq2Seq进行移位
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def get_dataset(self, train_dataset, dev_dataset, remove_columns: list):
        # 定义转换器
        trans_func = self._convert_example
        # trans_func = partial(self._convert_example,
        #                     text_column=self.prepare_config.text_column,
        #                     summary_column=self.prepare_config.target_column,
        #                     tokenizer=self.model_config.tokenizer,
        #                     max_source_length=self.prepare_config.max_source_length,
        #                     max_target_length=self.prepare_config.max_target_length)
                            
        # train_dataset和dev_dataset分别转换
        train_dataset = train_dataset.map(trans_func,
                                        batched=True,
                                        load_from_cache_file=True,
                                        remove_columns=remove_columns)
        dev_dataset = dev_dataset.map(trans_func,
                                    batched=True,
                                    load_from_cache_file=True,
                                    remove_columns=remove_columns)

        return train_dataset, dev_dataset
    
    def get_dataloader(self, train_dataset, dev_dataset):
        """获取模型所需的batchfy的dataloader。
        """
        # 组装 Batch 数据 & Padding
        batchify_fn = DataCollatorForSeq2Seq(tokenizer=self.model_config.tokenizer, model=self.model_config.model)
        # 分布式批采样器，用于多卡分布式训练
        train_batch_sampler = DistributedBatchSampler(
            train_dataset, batch_size=self.prepare_config.batch_size, shuffle=True)

        # 构造训练Dataloader
        train_data_loader = DataLoader(dataset=train_dataset,
                                    batch_sampler=train_batch_sampler,
                                    num_workers=0,
                                    collate_fn=batchify_fn,
                                    return_list=True)

        dev_batch_sampler = BatchSampler(dev_dataset,
                                        batch_size=self.prepare_config.batch_size,
                                        shuffle=False)
        # 构造验证Dataloader
        dev_data_loader = DataLoader(dataset=dev_dataset,
                                    batch_sampler=dev_batch_sampler,
                                    num_workers=0,
                                    collate_fn=batchify_fn,
                                    return_list=True)

        return train_data_loader, dev_data_loader
