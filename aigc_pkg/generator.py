"""Generatr to generate taget abstract or title.
"""

import os
import abc
import time
# import contextlib

from tqdm import tqdm
import numpy as np
import paddle
# import paddle.nn as nn
# from paddlenlp.transformers import PegasusForConditionalGeneration, PegasusChineseTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.metrics import BLEU
# from paddlenlp.data import DataCollatorForSeq2Seq
from visualdl import LogWriter
from rouge import Rouge

import prepare

class AbstractGeneratorConfig:
    def __init__(self, warmup, learning_rate,num_epochs,adam_epsilon,weight_decay,log_steps,eval_steps, 
    num_beams, min_target_length,output_dir= 'checkpoints', log_dir='visualdl_log_dir'):
        
        # 学习率预热比例
        self.warmup  = warmup
        # 学习率
        self.learning_rate  = learning_rate
        # 训练轮次
        self.num_epochs  = num_epochs
        # AdamW优化器参数epsilon
        self.adam_epsilon  = adam_epsilon
        # AdamW优化器参数weight_decay
        self.weight_decay  = weight_decay
        # 训练中，每个log_steps打印一次日志
        self.log_steps  = log_steps
        # 训练中，每隔eval_steps进行一次模型评估
        self.eval_steps  = eval_steps
        
        # 解码beam size
        self.num_beams  = num_beams
        # 摘要的最小长度
        self.min_target_length  = min_target_length

        # 训练模型保存路径
        self.output_dir = output_dir
        # 训练日志保存路径
        self.log_dir=log_dir
 
class BaseGenerator(abc.ABC):
    """Base class for preparation.
    """
    def __init__(self, train_data_loader, dev_data_loader, model_set: prepare.ModelSet, 
        prepare_config: prepare.AbstractGenerationPrapare,  gen_config: AbstractGeneratorConfig):
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.model_set = model_set
        self.prepare_config = prepare_config
        self.gen_config = gen_config
        self.optimizer, self.lr_scheduler, self.log_writer, self.decay_params, self.num_training_steps = self._get_train_tools()

    def _get_train_tools(self):
        """
        TODO: accept other lr_schedule or optimizer.
        """
        log_writer = LogWriter(self.gen_config.log_dir)
        # 训练总步数
        num_training_steps = len(self.train_data_loader) * self.gen_config.num_epochs
        lr_scheduler = LinearDecayWithWarmup(self.gen_config.learning_rate, num_training_steps, self.gen_config.warmup)

        # LayerNorm参数不参与weight_decay
        decay_params = [
            p.name for n, p in self.model_set.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        # 优化器AdamW
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            beta1=0.9,
            beta2=0.999,
            epsilon=self.gen_config.adam_epsilon,
            parameters=self.model_set.model.parameters(),
            weight_decay=self.gen_config.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)
        return optimizer, lr_scheduler, log_writer, decay_params, num_training_steps
    
    # 计算训练评估参数Rouge-1，Rouge-2，Rouge-L，BLEU-4
    def _compute_metrics(preds, targets):
        assert len(preds) == len(targets), (
            'The length of pred_responses should be equal to the length of '
            'target_responses. But received {} and {}.'.format(
                len(preds), len(targets)))
        rouge = Rouge()
        bleu4 = BLEU(n_size=4)
        scores = []
        for pred, target in zip(preds, targets):
            try:
                score = rouge.get_scores(' '.join(pred), ' '.join(target))
                scores.append([
                    score[0]['rouge-1']['f'], score[0]['rouge-2']['f'],
                    score[0]['rouge-l']['f']
                ])
            except ValueError:
                scores.append([0, 0, 0])
            bleu4.add_inst(pred, [target])
        rouge1 = np.mean([i[0] for i in scores])
        rouge2 = np.mean([i[1] for i in scores])
        rougel = np.mean([i[2] for i in scores])
        bleu4 = bleu4.score()
        print('\n' + '*' * 15)
        print('The auto evaluation result is:')
        print('rouge-1:', round(rouge1*100, 2))
        print('rouge-2:', round(rouge2*100, 2))
        print('rouge-L:', round(rougel*100, 2))
        print('BLEU-4:', round(bleu4*100, 2))
        return rouge1, rouge2, rougel, bleu4
    
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass
    
    @abc.abstractmethod
    def infer(self):
        pass

# TODO 完善train dev test
class AbstractGenerator(BaseGenerator):
    """Prepare data for NLG tasks abstract generation.
    """
    def __init__(self, train_data_loader, dev_data_loader,  model_set: prepare.ModelSet, 
        prepare_config: prepare.AbstractGenerationPrapare, gen_config: AbstractGeneratorConfig):
        super(AbstractGenerator, self).__init__(train_data_loader, dev_data_loader, model_set, prepare_config, gen_config)

    def train(self):
        global_step = 0
        best_rougel = 0
        tic_train = time.time()
        for epoch in range(self.gen_config.num_epochs):
            for step, batch in enumerate(self.train_data_loader):
                global_step += 1
                # 模型前向训练，计算loss
                _, _, loss = self.model_set.model(**batch)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.clear_grad()
                if global_step % self.gen_config.log_steps == 0:
                    logger.info(
                        "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                        % (global_step, self.num_training_steps, epoch, step,
                            paddle.distributed.get_rank(), loss, self.optimizer.get_lr(),
                            self.gen_config.log_steps / (time.time() - tic_train)))
                    self.log_writer.add_scalar("train_loss", loss.numpy(), global_step)
                    tic_train = time.time()
                if global_step % self.gen_config.eval_steps== 0 or global_step == self.num_training_steps:
                    tic_eval = time.time()
                    rouge1, rouge2, rougel, bleu4 = self.evaluate()
                    logger.info("eval done total : %s s" % (time.time() - tic_eval))
                    self.log_writer.add_scalar("eval_rouge1", rouge1, global_step)
                    self.log_writer.add_scalar("eval_rouge2", rouge2, global_step)
                    self.log_writer.add_scalar("eval_rougel", rougel, global_step)
                    self.log_writer.add_scalar("eval_bleu4", bleu4, global_step)
                    if best_rougel < rougel:
                        best_rougel = rougel
                        if paddle.distributed.get_rank() == 0:
                            if not os.path.exists(self.gen_config.output_dir):
                                os.makedirs(self.gen_config.output_dir)
                            # Need better way to get inner model of DataParallel
                            model_to_save = self.model_set.model._layers if isinstance(
                                self.model_set.model, paddle.DataParallel) else self.model_set.model
                            model_to_save.save_pretrained(self.gen_config.output_dir)
                            self.model_set.tokenizer.save_pretrained(self.gen_config.output_dir)
                

        
    @paddle.no_grad()
    def evaluate(self, data_loader):
        # use default dev data loader if not specified.
        data_loader = self.dev_data_loader

        self.model_set.model.eval()
        all_preds = []
        all_labels = []
        self.model_set.model = self.model_set.model._layers if isinstance(
            self.model_set.model, paddle.DataParallel) else self.model_set.model
        for batch in tqdm(data_loader, total=len(data_loader), desc="Eval step"):
            labels = batch.pop('labels').numpy()
            # 模型生成
            preds = self.model_set.model.generate(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                min_length=self.prepare_config.min_target_length,
                                max_length=self.prepare_config.max_target_length,
                                diversity_rate='beam_search',
                                num_beams=self.gen_config.num_beams,
                                use_cache=True)[0]
            # tokenizer将id转为string
            all_preds.extend(
                self.model_set.tokenizer.batch_decode(preds.numpy(),
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False))
            labels = np.where(labels != -100, labels, self.model_set.tokenizer.pad_token_id)
            all_labels.extend(
                self.model_set.tokenizer.batch_decode(labels,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False))
        rouge1, rouge2, rougel, bleu4 = self._compute_metrics(all_preds, all_labels)
        self.model_set.model.train()
        return rouge1, rouge2, rougel, bleu4
    
    def infer(self, text, text_column='input_ids', decode_strategy='beam_search'):
        tokenized = self.model_set.tokenizer(text, 
                          truncation=True, 
                          max_length=self.prepare_config.max_source_length, 
                          return_tensors='pd')
        preds, _ = self.model_set.model.generate(input_ids=tokenized[text_column],
                                max_length=self.prepare_config.max_target_length,
                                min_length=self.prepare_config.min_target_length,
                                decode_strategy=decode_strategy,
                                num_beams=self.gen_config.num_beams)
        return self.model_set.tokenizer.decode(preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)


