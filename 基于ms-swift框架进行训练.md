# 1. MS-Swift×Qwen3-VL

代码：https://github.com/modelscope/ms-swift

文档：[Swift DOCUMENTATION — swift 3.12.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/)

## 1.1 数据集准备

### 1.1.1 分析样例数据集

```python
{"id:", "conversations:[{"from":"human", "value":}, {"from":"gpt", "value":}]", "images":}
```

首先给我的是.parquet文件，所以要将.parquet文件进行解析并且格式转换

原数据格式：

```python
root_dir#图片文件根目录
image#图片名称
encrypt#图片是否加密
question#问题
ansewr#回答
dialogue_type#计数类型
```

接下来要写个脚本进行格式转换，写代码当然要借助于ai了，分享下我的prompt

```python
写一段代码将.parquet文件中的数据解析成json格式，json参考格式为{"id:", "conversations:[{"from":"human", "value":}, {"from":"gpt", "value":}]", "images":}，其中id对应于.parquet文件中的image去掉.jpg，human中的value对应于question，gpt中的value对应于answer，写一段代码
```

> [!WARNING]
>
> 值得注意的是，question和answer中的格式需要我们打印出来分析格式并转换

此外，我还遇到了一下几个问题：

1.图片路径改变：这种改变仅是某一个文件夹的名称改变，所以写一段代码将其替换就好了

2.训练的脚本用的是jsonl，所以还要写一段代码将json文件转换为jsonl文件就好了

以上内容是我自己探索的

### 1.1.2 现在讲一下顺利的工作内容：

提取.parquet文件的信息→坐标转换→构建对话内容→生成jsonl文件→写可视化脚本验证坐标转换的正确性

> [!NOTE]
>
> 1.在提取.parquet文件的信息中需要明确.parquet文件的数据格式以及如何提取。有意思的是，.parquet文件的question中格式为`物体1|**|物体2|**|`，answer格式为`[[]]|**|[[1,2;12,34]]|**|`，所以要将label和其坐标对应好。所以最终就是将原文件的单行数据拆解为多条对话。
>
> 2.在进行坐标转换后，记得进行可视化。在训练过程中，正负样本最好设置为1：3。
>
> 3.接下来就是构建对话内容，注意数据集的格式。
>
> 参考格式：
>
> ```python
> {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>描述图像"}, {"role": "assistant", "content": "<|object_ref_start|>一只狗<|object_ref_end|><|box_start|>(221,423),(569,886)<|box_end|>和<|object_ref_start|>一个女人<|object_ref_end|><|box_start|>(451,381),(733,793)<|box_end|>正在沙滩上玩耍"}], "images": ["/xxx/x.jpg"]}
> {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>找到图像中的<|object_ref_start|>羊<|object_ref_end|>"}, {"role": "assistant", "content": "<|box_start|>(101,201),(150,266)<|box_end|><|box_start|>(401,601),(550,666)<|box_end|>"}], "images": ["/xxx/x.jpg"]}
> {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>帮我打开谷歌浏览器"}, {"role": "assistant", "content": "Action: click(start_box='<|box_start|>(246,113)<|box_end|>')"}], "images": ["/xxx/x.jpg"]}
> ```
>
> 值得注意的是，
>
> - 不同模型grounding任务的特殊字符和数据集格式不同。
> - 不同模型对bbox是否归一化的处理不同。例如：qwen2.5-vl使用绝对坐标，而qwen2/3-vl、internvl2.5需要对bbox的坐标进行千分位坐标归一化。（这个问题前文已经处理过了）
>
> 重点：自己设计提示词的时候要用Qwen3-VL测试一下它能不能理解这个提示词
>
> 4.最后写个脚本验证坐标转换的正确性。

### 1.1.3 数据集注册

首先明确数据类型及格式，参考[自定义数据集 — swift 3.12.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html)

## 1.2 训练脚本

```python
#!/bin/bash
set -e

cd /data1/ms-swift

export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
# 有几个显卡就是几
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export USE_HF=False
export USE_MODELSCOPE=False

DATASET_PATHS=$(find 数据集路径 -name "*.jsonl" -type f | sort | tr '\n' ' ' | sed 's/,$//')

echo "使用数据集: $DATASET_PATHS"

#注意pt和sft的区别
swift sft \
    --model 自定义模型路径 \
    --model_type qwen3_vl \
    --dataset "$DATASET_PATHS" \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --max_length 4096 \
    --output_dir 输出路径 \
    --warmup_ratio 0.03 \
    --ddp_backend nccl \
    --attn_impl flash_attn \
    --packing true \
    --packing_length 8192 \
    --padding_free true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 10 \
    --eval_steps 1000 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --strict False \
    --deepspeed 有待考察 \
    --save_only_model true \

echo "模型将在以下目录输出: /data1/ms-swift/output_qwen3_vl_full_sft"
```

## 1.3 训练结果分析

### 1.3.1 对模型追加epoch后模型精度下降

1.resume_from_checkpoint不好使

那是因为设置了

```python
save_only_model True \
```

2.将model改成自己的检查点，那么模型的学习率是否该改变？

3.可能是数据的问题，检查一下数据是否转换正确

去掉content的负样本，让回答的token的长度不一致有利于模型学习

### 1.3.2 模型回答的截断

## 1.4 ms-swift源码解读

### 1.4.1 整体调用链

`SFT流程：swift sft → sft_main() → SwiftSft(args).main() → run() → train(trainer) → trainer.train(…)`

swift 可执行程序（console-script）
→ swift/cli/sft.py（CLI 子命令入口，基本就把参数解析后转给 sft_main）
→ swift/pipelines/train/sft.py:sft_main（核心业务：构造 SwiftSft/Trainer、加载模型&数据、开训）
→ swift/trainers/...（对 HF Transformers Trainer 的封装与混入）
→ 期间会调用 swift/llm/utils/model.py（取模型&tokenizer）、swift/llm/template/...（prompt 模板/打包）、swift/llm/dataset/...（数据读取与预处理）等

### 1.4.2 整体流程图

#### 命令入口

命令入口：`swift（console_scripts）`把 sft 子命令路由到 `swift/cli/sft.py`，该文件基本只负责解析/组合参数 → 调用 `sft_main(...)`。

```python
def try_init_unsloth():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuner_backend', type=str, default='peft')
    args, _ = parser.parse_known_args()
    if args.tuner_backend == 'unsloth':
        import unsloth


if __name__ == '__main__':
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    try_init_unsloth()
    from swift.ray import try_init_ray
    try_init_ray()
    from swift.pipelines import sft_main
    sft_main()#调用sft_main()
```

#### 主函数

主函数：`swift/pipelines/train/sft.py:sft_main`

组装/校验训练参数（集成参数 = 基础 + 原子参数），这套参数体系文档里有完整定义。(Swift 文档)
构造 SwiftSft(args) 并 main() → run() → train(trainer)。

```python
logger = get_logger()

@RayHelper.worker(group=['default'])
class SwiftSft(SwiftPipeline, TunerMixin):
    args_class = SftArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], SftArguments]] = None) -> None:
        super().__init__(args)
        self.train_msg = {}
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_flash_ckpt()

    @RayHelper.function(group='default')
    # Flash Checkpoint 准备
    def _prepare_flash_ckpt(self):
        if self.args.use_flash_ckpt:
            try:
                import dlrover.trainer.torch.flash_checkpoint.hf_trainer
            except ImportError:
                raise ValueError('Please install dlrover to use flash ckpt `pip install dlrover[k8s,torch]')

    def _prepare_generation_config(self):
        args = self.args
        self.model.origin_generation_config = self.model.generation_config
        self.model.generation_config = prepare_generation_config(self.model.generation_config,
                                                                 args.get_request_config(), self.tokenizer)
        logger.info(f'model.generation_config: {self.model.generation_config}')

    @RayHelper.function(group='default')
    def _prepare_model_tokenizer(self, **kwargs):
        args = self.args
        self.model, self.processor = args.get_model_processor(**kwargs)
        if args.sequence_parallel_size > 1:
            sequence_parallel.prepare(
                args.sequence_parallel_size, model=self.model, tokenizer=self.processor, padding_free=args.padding_free)
        if self.model is None:
            return
        if hasattr(self.model, 'hf_device_map'):
            logger.info(f'model.hf_device_map: {self.model.hf_device_map}')

        logger.info(f'model_info: {self.model.model_info}')

        self._prepare_generation_config()

    @RayHelper.function(group='default')
    def _prepare_template(self) -> None:
        args = self.args
        template = args.get_template(self.processor)
        template.set_mode('train')
        if template.use_model:
            template.model = self.model
        support_padding_free = template.support_padding_free
        if support_padding_free is None:
            support_padding_free = not args.model_meta.is_multimodal
        if (args.padding_free or args.packing) and not support_padding_free:
            raise ValueError(f'Template `{args.template}` does not support padding free or packing.')
        self.template = template

    def _get_dataset(self):
        # The random shuffling of the training set occurs in the dataloader of the trainer.
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = None, None
        if args.dataset:
            train_dataset, val_dataset = load_dataset(
                args.dataset,
                split_dataset_ratio=args.split_dataset_ratio,
                shuffle=args.dataset_shuffle,
                **dataset_kwargs)
        if len(args.val_dataset) > 0:
            # Loading val dataset
            dataset_kwargs.pop('interleave_prob', None)
            _, val_dataset = load_dataset(
                args.val_dataset, split_dataset_ratio=1.0, shuffle=args.val_dataset_shuffle, **dataset_kwargs)
            assert args.split_dataset_ratio == 0.
        if args.truncation_strategy == 'split':
            logger.info(f'train_dataset: {train_dataset}')
            logger.info(f'val_dataset: {val_dataset}')
        return train_dataset, val_dataset

    def _save_val_dataset(self, val_dataset):
        args = self.args
        output_dir = getattr(args, 'output_dir', None) or getattr(args, 'save')
        if is_master() and isinstance(val_dataset, HfDataset) and not args.val_dataset:
            os.makedirs(output_dir, exist_ok=True)
            val_dataset_path = os.path.join(output_dir, 'val_dataset.jsonl')
            append_to_jsonl(val_dataset_path, val_dataset.to_list())
            logger.info(f'The split dataset from the training set will be saved at: {val_dataset_path}.')

    @RayHelper.function(group='default')
    def _prepare_dataset(self):
        args = self.args
        # Defer encoding to the training phase
        pre_process = not (hasattr(args, 'rlhf_type') and args.rlhf_type in ['grpo', 'gkd'])
        if args.cached_dataset or args.cached_val_dataset:
            assert not args.streaming, 'Cached dataset does not support streaming.'
            train_datasets, val_datasets = get_cached_dataset(self.args)
        else:
            train_datasets, val_datasets = [], []
        if args.dataset or args.val_dataset:
            train_dataset, val_dataset = self._get_dataset()
            # 编码
            train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)
            if train_dataset is not None:
                train_datasets.append(train_dataset)
            if val_dataset is not None:
                val_datasets.append(val_dataset)
        train_dataset = DatasetLoader.concat_datasets(train_datasets)
        val_dataset = DatasetLoader.concat_datasets(val_datasets)
        if args.truncation_strategy != 'split':
            logger.info(f'train_dataset: {train_dataset}')
            logger.info(f'val_dataset: {val_dataset}')
        datasets = [train_dataset, val_dataset]
        if not pre_process:
            return datasets
        # 后处理
        datasets = self._post_process_datasets(datasets)
        self._show_dataset(*datasets)
        return datasets

    def _post_process_datasets(self, datasets: List) -> List:
        args = self.args
        predict_with_generate = getattr(args, 'predict_with_generate', False)

        template = self.template
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            if i == 1 and predict_with_generate:
                # val_dataset
                continue
            if not args.streaming and args.truncation_strategy != 'split':
                dataset = LazyLLMDataset(dataset, template.encode, strict=args.strict, random_state=args.data_seed)
            if args.packing:
                packing_dataset_cls = IterablePackingDataset if args.streaming else PackingDataset
                dataset = packing_dataset_cls(
                    template,
                    dataset,
                    num_proc=args.dataset_num_proc,
                    packing_length=args.packing_length,
                    packing_num_proc=args.packing_num_proc,
                    strict=args.strict,
                    load_from_cache_file=args.load_from_cache_file)
            elif args.streaming:
                preprocessor = EncodePreprocessor(template=template)
                dataset = preprocessor(
                    dataset,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=args.load_from_cache_file,
                    strict=args.strict)
            datasets[i] = dataset
        return datasets

    @RayHelper.function(group='default')
    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()

        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        args.save_args()

        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            template=self.template,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)

    def _get_trainer_kwargs(self):
        return {}

    def _save_trainer_state(self, trainer):
        training_args = trainer.args
        state = trainer.state
        if hasattr(state, 'last_model_checkpoint'):
            if self.args.create_checkpoint_symlink:
                last_checkpoint = os.path.join(self.args.output_dir, 'last')
                best_checkpoint = os.path.join(self.args.output_dir, 'best')
                if is_master():
                    os.symlink(state.last_model_checkpoint, last_checkpoint)
                    os.symlink(state.best_model_checkpoint, best_checkpoint)
                state.last_model_checkpoint = last_checkpoint
                state.best_model_checkpoint = best_checkpoint
        else:
            state.last_model_checkpoint = None
        logger.info(f'last_model_checkpoint: {state.last_model_checkpoint}')
        logger.info(f'best_model_checkpoint: {state.best_model_checkpoint}')

        # Visualization
        if is_master():
            if 'tensorboard' in training_args.report_to:
                images_dir = os.path.join(training_args.output_dir, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)
            if training_args.push_to_hub:
                trainer.push_to_hub()

        self.train_msg.update({
            'last_model_checkpoint': state.last_model_checkpoint,
            'best_model_checkpoint': state.best_model_checkpoint,
            'best_metric': state.best_metric,
            'global_step': state.global_step,
            'log_history': state.log_history,
            'memory': getattr(state, 'max_memory', None),
        })
        if is_master():
            jsonl_path = os.path.join(training_args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, self.train_msg, strict=False)
        return self.train_msg

    def train(self, trainer):
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        try:

            resume_checkpoint = None
            callbacks = set(getattr(self.args, 'callbacks', []))
            elastic_enabled = 'deepspeed_elastic' in callbacks
            # If flash checkpoint is enabled, try to resume from the last complete checkpoint.
            # If the previous training finished, resume_checkpoint stays None.
            if self.args.use_flash_ckpt:
                # resume_checkpoint = <resume_dir>/checkpoint-<step>
                resume_checkpoint = trainer.get_resume_checkpoint()
            # Elastic runs require a universal checkpoint; fall back when missing or incomplete.
            if elastic_enabled and (resume_checkpoint is None
                                    or not os.path.exists(os.path.join(resume_checkpoint, 'latest_universal'))):
                # get_resume_checkpoint_until_find_ucp returns <resume_dir>/checkpoint-<step> with latest_universal,
                # or None; when None, no universal checkpoint exists and training starts from scratch.
                resume_checkpoint = trainer.get_resume_checkpoint_until_find_ucp()
            # Explicit user override always wins.
            if self.args.resume_from_checkpoint:
                resume_checkpoint = self.args.resume_from_checkpoint
            trainer.train(resume_checkpoint)
        finally:
            res = self._save_trainer_state(trainer)
            if self.args.use_flash_ckpt and hasattr(trainer, 'flash_checkpointer'):
                trainer.wait_latest_checkpoint(trainer.FLASH_CKPT_WAIT_TIMEOUT, trainer.state.global_step)

        return res

    @staticmethod
    # 统计数据集信息
    def _stat_dataset(dataset: Union[HfDataset, PackingDataset, LazyLLMDataset]):
        if isinstance(dataset, LazyLLMDataset):
            dataset = dataset.dataset
        if isinstance(dataset, HfDataset):
            lengths = dataset['lengths']
            lengths = [max(length) if isinstance(length, list) else length for length in lengths]
        else:
            lengths = dataset.packed_length
        _, stat_str = stat_array(lengths)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str

    def _show_dataset(self, train_dataset, val_dataset):
        args = self.args
        predict_with_generate = getattr(args, 'predict_with_generate', False)
        if is_master():
            inputs = train_dataset[0] if hasattr(train_dataset, '__len__') else next(iter(train_dataset))
            if isinstance(inputs, list):
                inputs = inputs[0]
            self.template.print_inputs(inputs)
        elif hasattr(train_dataset, '__len__'):
            # Avoid the random mismatch issue in LazyLLMDataset.
            inputs = train_dataset[0]
        if val_dataset is not None and hasattr(val_dataset, '__len__') and len(val_dataset) == 0:
            val_dataset = None
        if not args.lazy_tokenize and not args.streaming:
            self.train_msg['train_dataset'] = self._stat_dataset(train_dataset)
            if val_dataset is not None and not predict_with_generate:
                self.train_msg['val_dataset'] = self._stat_dataset(val_dataset)

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        template = self.template
        args = self.args
        self._save_val_dataset(val_dataset)

        predict_with_generate = getattr(args, 'predict_with_generate', False)
        datasets = [train_dataset, val_dataset]
        if not pre_process:
            return datasets

        origin_template_model = template.model
        template.model = None  # Avoid serializing the model.
        if args.truncation_strategy == 'split':
            if (args.task_type != 'causal_lm' or template.mode != 'train' or args.use_chat_template
                    or args.model_meta.is_multimodal):
                raise ValueError(
                    '`--truncation_strategy split` is currently only supported for plain text model pretraining')
            assert not args.lazy_tokenize, '`--truncation_strategy split` does not support lazy_tokenize'

        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            if i == 1 and predict_with_generate:
                # val_dataset
                continue
            if not args.lazy_tokenize and not args.streaming:
                # Compatible with cached_dataset, only additionally write length here.
                preprocessor_cls = EncodePreprocessor if args.truncation_strategy == 'split' else AddLengthPreprocessor
                preprocessor = preprocessor_cls(template=template)
                batch_size = 100 if args.model_meta.is_multimodal else 1000
                dataset = preprocessor(
                    dataset,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=args.load_from_cache_file,
                    strict=args.strict,
                    batch_size=batch_size)
            datasets[i] = dataset
        template.model = origin_template_model

        return datasets


def sft_main(args: Optional[Union[List[str], SftArguments]] = None):
    return SwiftSft(args).main()
```

#### 模型/分布式/LoRA 等准备

swift.llm.utils.model.get_model_tokenizer(...)：下载/加载 base 模型、按 --train_type 注入 LoRA/QLoRA/DoRA 等、设置 dtype、并行策略（FSDP/DeepSpeed/Megatron 等）、注意力内核（flash/SDPA）等。官方 CLI 参数文档也说明了这些开关。

#### 数据与模板

swift.llm.template.template.：按 --template 或模型默认模板把多轮对话打包（packing、特殊 token、role 映射等）。
swift.llm.dataset.*：按 --dataset 或本地 JSONL/Parquet 加载并过滤、tokenize、collate。

#### 训练执行

训练执行：

swift.trainers.*：在 HF Trainer 基础上加了混入（如 swift/trainers/mixin.py），封装了 resume/eval/logging/callback 等。回溯能看到 trainers/mixin.py 参与。(GitHub)

#### 保存/合并/导出

训练结束保存 adapter/checkpoint；按需 --merge_lora true 合并权重；导出给 vLLM/SGLang/LmDeploy 推理。


## 附录

### Qwen3-VL Grounding数据格式

来源：[Qwen3-VL/cookbooks/2d_grounding.ipynb at main · QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)

一共有三类任务：Boxed-based Grounding、Point-based Grounding以及Counting。

与Qwen2.5-VL不同的是，该版本采用了一个归一化坐标系，其范围为[0，1000]。

#### 数据格式

https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb

#### 坐标转换

Qwen3的坐标系统已经从Qwen2.5-VL使用的绝对坐标更改为从0到1000的相对坐标

```python
# 从相对坐标改为绝对坐标
# Convert normalized coordinates to absolute coordinates
abs_y1 = int(bounding_box["bbox_2d"][1] / 1000 * height)
abs_x1 = int(bounding_box["bbox_2d"][0] / 1000 * width)
abs_y2 = int(bounding_box["bbox_2d"][3] / 1000 * height)
abs_x2 = int(bounding_box["bbox_2d"][2] / 1000 * width)
```

`int()` 函数通过向下取整来简化计算，可能会牺牲一些精度，特别是在处理接近边界的坐标时；而 `round()` 函数通过四舍五入来提供更精确的坐标，有助于减少误差，特别是在需要精确定位时。

以这种坐标算法为例，研究一下怎么进行坐标转换的

```python
x_1 = min(max(round(x_1/max(w,h)*999),0),999)
```

首先将 `x_1` 转换为相对于图像最大维度的比例，并乘以999；然后对结果进行四舍五入，以得到一个整数。最后确保结果不小于0且不大于999。

转换代码

```python
首先将[0,999]归一化到像素基准
x_1=round(x_1*max(w,h)/999)
然后变成Qwen3-VL坐标
x_1=round(x_1/w*1000)
```

### 预训练

预训练使用`swift pt`命令，这将自动使用生成式而非对话式的template，即将`use_chat_template`设置为False（其他所有的命令，例如`swift sft/rlhf/infer`，都默认将`use_chat_template`设置为True）。此外，`swift pt`与`swift sft`相比，具有不同的数据集格式。

自定义的数据集接入有三种方法，这里采用直接使用命令行传参的方式接入，即`--dataset <dataset_path1> <dataset_path2>`。这将使用AutoPreprocessor将数据集转换为标准格式（支持4种数据集格式，具体查看下面对AutoPreprocessor的介绍）。你可以使用`--columns`进行列名转换。支持传入csv、json、jsonl、txt、文件夹（例如git clone开源数据集）。该方案不需要修改dataset_info.json，适合刚接触ms-swift的用户。

### 参数分析

- 🔥freeze_llm: 该参数只对多模态模型生效，可用于全参数训练和LoRA训练，但会产生不同的效果。若是全参数训练，将freeze_llm设置为True会将LLM部分权重进行冻结；若是LoRA训练且`target_modules`设置为'all-linear'，将freeze_llm设置为True将会取消在LLM部分添加LoRA模块。该参数默认为False。
- 🔥freeze_vit: 该参数只对多模态模型生效，可用于全参数训练和LoRA训练，但会产生不同的效果。若是全参数训练，将freeze_vit设置为True会将vit部分权重进行冻结；若是LoRA训练且`target_modules`设置为'all-linear'，将freeze_vit设置为True将会取消在vit部分添加LoRA模块。该参数默认为True。
  - 注意：**这里的vit不仅限于vision_tower, 也包括audio_tower**。若是Omni模型，若你只希望对vision_tower加LoRA，而不希望对audio_tower加LoRA，你可以修改[这里的代码](https://github.com/modelscope/ms-swift/blob/a5d4c0a2ce0658cef8332d6c0fa619a52afa26ff/swift/llm/model/model_arch.py#L544-L554)。
- 🔥freeze_aligner: 该参数只对多模态模型生效，可用于全参数训练和LoRA训练，但会产生不同的效果。若是全参数训练，将freeze_aligner设置为True会将aligner（也称为projector）部分权重进行冻结；若是LoRA训练且`target_modules`设置为'all-linear'，将freeze_aligner设置为True将会取消在aligner部分添加LoRA模块。该参数默认为True。
- 🔥interleave_prob：默认值为 None。在组合多个数据集时，默认使用datasets库的 `concatenate_datasets` 函数；如果设置了该参数，则会使用 `interleave_datasets` 函数。该参数通常用于流式数据集的组合，并会作为参数传入 `interleave_datasets` 函数中。该参数不对`--val_dataset`生效。当使用多个数据集时，可以支持不同数据集按照比例采样（根据文档目前只支持按照数量采样），对于预训练任务来说是个很常用的功能。

部分常用参数说明：https://developer.aliyun.com/article/1668259

sft常用参数

```python
--model_type: 表示你选择的模型类型, 默认是None. model_type指定了对应模型默认的lora_target_modules, template_type等信息. 你可以通过只指定model_type进行微调. 对应的model_id_or_path会使用默认的设置, 从ModelScope进行下载, 并使用默认的缓存路径. model_type和model_id_or_path必须指定其中的一个. 可以选择的model_type可以查看支持的模型.
--model_id_or_path: 表示模型在ModelScope Hub中的model_id或者本地路径, 默认为None. 如果传入的model_id_or_path已经被注册, 则会根据model_id_or_path推断出model_type. 如果未被注册, 则需要同时指定model_type, e.g. --model_type <model_type> --model_id_or_path <model_id_or_path>.
--model_revision: 表示模型在ModelScope Hub中对应model_id的版本号, 默认为None. model_revision指定为None, 则使用注册在MODEL_MAPPING中的revision. 否则强制使用命令行传入的model_revision.
--sft_type: 表示微调的方式, 默认是'lora'. 你可以选择的值包括: 'lora', 'full', 'longlora', 'qalora'. 如果你要使用qlora, 你需设置--sft_type lora --quantization_bit 4.
--freeze_parameters: 当sft_type指定为'full'时, 将模型最底部的参数进行freeze. 指定范围为0. ~ 1., 默认为0.. 该参数提供了lora与全参数微调的折中方案.
--additional_trainable_parameters: 作为freeze_parameters的补充, 只有在sft_type指定为'full'才允许被使用, 默认为[]. 例如你如果想训练50%的参数的情况下想额外训练embedding层, 你可以设置--freeze_parameters 0.5 --additional_trainable_parameters transformer.wte, 所有以transformer.wte开头的parameters都会被激活.
--tuner_backend: 表示lora, qlora的后端支持, 默认是'swift'. 你可以选择的值包括: 'swift', 'peft'.
--template_type: 表示使用的对话模板的类型, 默认是'AUTO', 即根据model_type查找MODEL_MAPPING中的template. 可以选择的template_type可以查看TEMPLATE_MAPPING.keys().
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
--output_dir: 表示ckpt存储的目录, 默认是'output'. 我们会在该目录后拼接model_type和微调版本号. 方便用户对不同模型进行多次对比实验, 而不需要改变output_dir命令行参数. 如果不需要拼接这些内容, 你需要额外指定参数--add_output_dir_suffix false.
--add_output_dir_suffix: 默认为True, 表示会在output_dir的目录后拼接上model_type和微调版本号的后缀. 如果要避免此行为, 你可以设置为False.
--ddp_backend: 表示分布式的后端支持, 默认是'nccl'. 你可以选择的值包括: 'nccl', 'gloo', 'mpi', 'ccl'.
--seed: 全局的seed, 默认使用42. 用于复现训练效果.
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
--resume_from_checkpoint: 用于断点续训, 默认为None. 你可以将其设置为checkpoint的路径, 例如: 'output/qwen-7b-chat/vx-xxx/checkpoint-xxx', 来进行断点续训.
--dtype: 基模型载入时的torch_dtype, 默认为'AUTO', 即智能选择dtype: 如果机器不支持bf16, 则使用fp16, 如果MODEL_MAPPING中对应模型有指定torch_dtype, 则使用其对应dtype, 否则使用bf16. 你可以选择的值包括: 'bf16', 'fp16', 'fp32'.
--dataset: 用于选择训练的数据集, 默认为[]. 可以选择的数据集可以查看支持的数据集. 如果需要使用多个数据集进行训练, 你可以使用','或者' '进行分割, 例如: --dataset alpaca-en,alpaca-zh or --dataset alpaca-en alpaca-zh.
--dataset_seed: 用于指定数据集处理的seed, 默认为42. 以random_state形式存在, 不影响全局seed.
--dataset_test_ratio: 用于指定子数据集切分成训练集和验证集的比例, 默认为0.01. 如果子数据集已经进行了训练集和验证集的切分, 则此参数无效.
--train_dataset_sample: 对训练集进行采样, 默认是20000, 用于加快训练的速度. 该参数是为了避免数据集过大, 单个epoch训练时间过长的问题. 如果你指定为-1, 则使用完整的训练集进行训练.
--val_dataset_sample: 对验证集进行采样, 默认是None, 自动选取合适数量的数据集数量进行验证. 如果你指定为-1, 则使用完整的验证集进行验证.
--system: 对话模板中使用的system, 默认为None, 即使用模型默认的system. 如果指定为'', 则不使用system.
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
--max_length: token的最大长度, 默认为2048. 可以避免个别过长的数据样本造成OOM的问题. 当指定--truncation_strategy delete时, 如果某数据样本长度超过max_length, 我们会删除该数据样本. 如果指定--truncation_strategy truncation_left时, 我们会切除最前面的token: input_ids[-max_length:]. 如果设置为-1, 则无限制.
--truncation_strategy: 默认是'delete'表示把超过max_length的句子从数据集中删除. 'truncation_left'表示会将超过文本的左边给切除掉, 这可能会切到special token, 会影响性能, 并不推荐.
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
--check_dataset_strategy: 默认值为'none', 即不做检查. 如果你训练的模型是LLM, 则推荐使用'warning'作为数据检查的策略. 如果你的训练目标为句子分类等任务, 则建议设置为'none'.
--custom_train_dataset_path: 默认值为[]. 具体的含义参考自定义与拓展.
--custom_val_dataset_path: 默认值为[]. 具体的含义参考自定义与拓展.
--self_cognition_sample: 自我认知数据集的采样数. 默认为0. 你该值设置为>0时, 需要同时指定--model_name, --model_author. 如果你想了解更多, 可以查看自我认知微调最佳实践.
--model_name: 默认为[None, None]. 如果开启了自我认知数据集的采样(即self_cognition_sample>0), 你需要传入两个值, 分别代表模型的中文名和英文名. 例如: --model_name 小黄 'Xiao Huang'.
--model_author: 默认为[None, None]. 如果开启了自我认知数据集的采样, 你需要传入两个值, 分别代表作者的中文名和英文名. 例如: --model_author 魔搭 ModelScope.
--quantization_bit: 用于指定是否进行量化和量化的bit数, 默认为0, 即不进行量化. 如果要使用4bit qlora, 你需要设置--sft_type lora --quantization_bit 4
--bnb_4bit_comp_dtype: 在进行4bit量化时, 我们需要在模型的forward和backward时, 将其进行反量化. 该参数用于指定反量化后的torch_dtype. 默认为'AUTO', 即与dtype保持一致. 可选择的值包括: 'fp16', 'bf16', 'fp32'. 当quantization_bit为0时, 该参数无效.
--bnb_4bit_quant_type: 4bit量化时的量化方式, 默认是'nf4'. 可选择的值包括: 'nf4', 'fp4'. 当quantization_bit为0时, 该参数无效.
--bnb_4bit_use_double_quant: 是否在4bit量化时开启double量化, 默认为True. 当quantization_bit为0时, 该参数无效.
--lora_target_modules: 指定lora模块, 默认为['DEFAULT']. 如果lora_target_modules传入'DEFAULT' or 'AUTO', 则根据model_type查找MODEL_MAPPING中的lora_target_modules(默认指定为qkv). 如果传入'ALL', 则将所有的Linear层(不含head)指定为lora模块. 如果传入'EMBEDDING', 则Embedding层指定为lora模块. 如果内存允许, 建议设置成'ALL'. 当然, 你也可以设置['ALL', 'EMBEDDING'], 将所有的Linear和embedding层指定为lora模块. 该参数只有当sft_type指定为'lora'时才生效.
--lora_rank: 默认为8. 只有当sft_type指定为'lora'时才生效.
--lora_alpha: 默认为32. 只有当sft_type指定为'lora'时才生效.
--lora_dropout_p: 默认为0.05, 只有当sft_type指定为'lora'时才生效.
--lora_bias_trainable: 默认为'none', 可以选择的值: 'none', 'all'. 如果你要将bias全都设置为可训练, 你可以设置为'all'.
--lora_modules_to_save: 默认为[]. 如果你想要训练embedding, lm_head, 或者layer_norm, 你可以设置此参数, 例如: --lora_modules_to_save EMBEDDING LN lm_head. 如果传入'EMBEDDING', 则将Embedding层添加到lora_modules_to_save. 如果传入'LN', 则将RMSNorm和LayerNorm添加到lora_modules_to_save.
--lora_dtype: 默认为'fp32', 指定lora模块的dtype类型. 如果是AUTO则跟随原始模块的dtype类型. 你可以选择的值: 'fp16', 'bf16', 'fp32', 'AUTO'.
--use_dora: 默认为False, 是否使用DoRA.
--use_rslora: 默认为False, 是否使用RS-LoRA.
--neftune_noise_alpha: NEFTune添加的噪声系数, 可以提升模型在指令微调中的性能, 默认为None. 通常可以设置为5, 10, 15. 你可以查看相关论文.
--neftune_backend: NEFTune的backend，默认使用transformers库, 当训练VL模型时可能遇到不适配的情况, 此时建议指定为swift.
--gradient_checkpointing: 是否开启gradient checkpointing, 默认为True. 该参数可以用于节约显存, 虽然这会略微降低训练速度. 该参数在max_length较大, batch_size较大时作用显著.
--deepspeed: 用于指定deepspeed的配置文件的路径或者直接传入json格式的配置信息, 默认为None, 即不开启deepspeed. deepspeed可以节约显存. 我们书写了默认的ZeRO-2配置文件, ZeRO-3配置文件. 你只需要指定'default-zero2', 就会使用默认zero2配置文件; 指定'default-zero3', 就会使用默认的zero3配置文件.
--batch_size: 训练时的batch_size, 默认为1. 增大batch_size可以增加GPU的利用率, 但不一定会增加训练速度, 因为在一个batch中, 需要对较短的句子按该batch中最长句子的长度进行padding, 从而引入无效的计算量.
--eval_batch_size: 评估时的batch_size, 默认为None, 即当predict_with_generate为True时, 设置为1, 为False时, 设置为batch_size.
--num_train_epochs: 训练的epoch数, 默认为1. 如果max_steps >= 0, 则覆盖num_train_epochs. 通常情况下设置为3 ~ 5.
--max_steps: 训练的max_steps数, 默认为-1. 如果max_steps >= 0, 则覆盖num_train_epochs.
--optim: 默认为'adamw_torch'.
--learning_rate: 默认值为None, 即如果sft_type为lora, 则设置为1e-4, 如果sft_type为full, 则设置为1e-5.
--weight_decay: 默认值为0.01.
--gradient_accumulation_steps: 梯度累加, 默认值为None, 设置为math.ceil(16 / self.batch_size / world_size). total_batch_size = batch_size * gradient_accumulation_steps * world_size.
--max_grad_norm: 梯度裁剪, 默认值为0.5.
--predict_with_generate: 评估时是否使用生成式的方式, 默认为False. 如果设置为False, 则使用loss进行评估. 如果设置为True, 则使用ROUGE-L等指标进行评估. 使用生成式评估耗费的时间很长, 请谨慎选择.
--lr_scheduler_type: 默认值为'linear', 你可以选择: 'linear', 'cosine', 'constant'等.
--warmup_ratio: warmup占用总的训练steps的比例, 默认为0.05.
--eval_steps: 每训练多少steps进行评估, 默认为50.
--save_steps: 每训练多少个steps进行保存, 默认为None, 即设置为eval_steps.
--save_only_model: 是否只保存模型参数, 而不存储断点续训所需的中间状态, 默认为None, 即如果sft_type为'lora'并且不使用deepspeed(deepspeed为None), 设置为False, 否则设置为True(e.g. 使用了全参数微调或者使用了deepspeed).
--save_total_limit: 保存的checkpoint的数量, 默认为2, 即保存best和last的checkpoint. 如果设置为-1, 则保存所有的checkpoint.
--logging_steps: 每训练多少步打印训练信息(e.g. loss, learning_rate等), 默认为5.
--dataloader_num_workers: 默认值为1.
--push_to_hub: 是否将训练的checkpoint同步推送到ModelScope Hub中, 默认为False.
--hub_model_id: 推送到的ModelScope Hub的model_id, 默认为None, 即设置为f'{model_type}-{sft_type}'. 你可以将其设置为model_id, 也可以设置为repo_name. 我们会根据hub_token推断出user_name. 推送的远程仓库如果不存在, 则会创建一个新的仓库, 如果存在, 则复用之前的仓库. 该参数只有在push_to_hub设置为True时才生效.
--hub_token: 推送时需要的SDK token. 可以从魔搭社区获取, 默认为None, 即从环境变量MODELSCOPE_API_TOKEN中获取. 该参数只有在push_to_hub设置为True时才生效.
--hub_private_repo: 推送的ModelScope Hub中的模型仓库的权限是否设置为私有, 默认为False. 该参数只有在push_to_hub设置为True时才生效.
--push_hub_strategy: 推送策略, 默认为'push_best'. 可选择的值包括: 'end', 'push_best', 'push_last', 'checkpoint', 'all_checkpoints'. 'push_best'表示在每次保存权重时, 将最好的模型进行推送并覆盖之前的权重, 'push_last'表示在每次保存权重时, 将最后的权重进行推送并覆盖之前的权重, 'end'表示只在训练的最后推送最好的模型. 该参数只有在push_to_hub设置为True时才生效.
--test_oom_error: 用于检测训练是否会发生OOM, 默认为False. 如果设置为True, 则会将训练集按max_length倒序进行排列, 方便OOM的测试. 该参数一般用于测试, 请谨慎设置.
--disable_tqdm: 是否不启用tqdm, 这在nohup启动脚本时很有用. 默认为False, 即为启动tqdm.
--lazy_tokenize: 如果设置为False, 则在trainer.train()之前提前对所有文本进行预处理. 如果设置为True, 则延迟对文本进行编码, 减少预处理的等待并减少内存占用, 这在处理大数据集时很有用. 默认为None, 即我们会根据template的类型进行智能选择, LLM的模型通常设置为False, 多模态的模型通常设置为True(避免图片和音频加载导致过多的内存占用).
--preprocess_num_proc: 在对数据集预处理时(对文本进行tokenize), 使用多进程. 默认为1. 与lazy_tokenize命令行参数一样, 用于解决预处理速度慢的问题. 但该策略无法减少内存占用, 所以如果当数据集巨大时, 建议使用lazy_tokenize. 推荐设置的值: 4, 8. 请注意: 当使用qwen-audio时, 该参数会强制设置为1, 因为qwen-audio的预处理函数中使用了torch的多进程, 会造成不兼容问题.
--use_flash_attn: 是否使用flash attn, 默认为None. 安装flash_attn的步骤可以查看https://github.com/Dao-AILab/flash-attention. 支持flash_attn的模型可以查看LLM支持的模型.
--ignore_args_error: 是否忽略命令行传参错误抛出的Error, 默认为False. 如果需要拷贝代码到notebook中运行, 需要设置成True.
--check_model_is_latest: 检查模型是否是最新, 默认为True. 如果你需要断网进行训练, 请将该参数设置为False.
--logging_dir: 默认为None. 即设置为f'{self.output_dir}/runs', 表示tensorboard文件存储路径.
--report_to: 默认为['tensorboard'].
--acc_strategy: 默认为'token', 可选择的值包括: 'token', 'sentence'.
--save_on_each_node: 该参数在多机训练时生效, 默认为True.
--save_strategy: 保存checkpoint的策略, 默认为'steps', 可选择的值包括: 'steps', 'no'.
--save_safetensors: 默认为True.
--max_new_tokens: 默认为2048. 该参数只有在predict_with_generate设置为True的时候才生效.
--do_sample: 默认为True. 该参数只有在predict_with_generate设置为True的时候才生效.
--temperature: 默认为0.3. 该参数只有在predict_with_generate设置为True的时候才生效.
--top_k: 默认为20. 该参数只有在predict_with_generate设置为True的时候才生效.
--top_p: 默认为0.7. 该参数只有在predict_with_generate设置为True的时候才生效.
--repetition_penalty: 默认为1.. 该参数只有在predict_with_generate设置为True的时候才生效.
--num_beams: 默认为1. 该参数只有在predict_with_generate设置为True的时候才生效.
--gpu_memory_fraction: 默认为None. 该参数旨在指定显卡最大可用显存比例的情况下运行训练，用于极限测试.
--train_dataset_mix_ratio: 默认为0. 该参数定义了如何进行数据集打混训练. 指定该参数时, 训练集会以train_dataset_mix_ratio倍数混合train_dataset_mix_ds指定的通用知识数据集, 使整体数据集长度达到train_dataset_sample.
--train_dataset_mix_ds: 默认为ms-bench. 用于防止知识遗忘的通用知识数据集.
--use_loss_scale: 默认为False. 生效时会将Agent的部分字段(Action/Action Input部分)的loss权重加强以强化CoT, 对普通SFT场景没有任何效果.
```



### 问题

#### 训练的时候CUDA out of memory

试试deepspeed

#### 断点续训

值得注意的是，在开始训模型的时候要将save_only_model去掉才可以保存优化器状态等文件

```python
--resume_from_checkpoint
# 例如
--resume_from_checkpoint /HDD/train/output/checkpoint-68086/v9-20250420-214017/checkpoint-2000 \
```

# 2. OSS_VLMEVALKIT框架评测模型

网站：https://github.com/open-compass/VLMEvalKit

## 2.1 模型路径修改

https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py

```python
"Qwen3-VL-4B-Instruct": partial(
    Qwen3VLChat,
    # model_path改成自己训练的模型路径
    model_path="Qwen/Qwen3-VL-4B-Instruct",
    use_custom_prompt=False,
    use_vllm=True,
    temperature=0.7, 
    max_new_tokens=16384,
    repetition_penalty=1.0,
    presence_penalty=1.5,
    top_p=0.8,
    top_k=20
),
```

确保在开始评估之前，你可以成功使用 VLM 进行推理，使用以下命令 `vlmutil check {MODEL_NAME}`

## 2.2 配置系统

默认情况下，VLMEvalKit通过在`run.py`脚本中使用`--model`和`--data`参数设置模型名称（在`/vlmeval/config.py`中定义）和数据集名称（在`vlmeval/dataset/__init__.py` 或 `vlmeval/dataset/video_dataset_config.py` 中定义）来启动评估。

当用户希望使用不同设置评估多个模型/数据集时，VLMEvalKit提供了一个更灵活的配置系统。用户可以在json文件中指定模型和数据集设置，并通过`--config`参数将配置文件的路径传递给`run.py`脚本。

```python
{
    "model": {
        "GPT4o_20240806_T00_HIGH": {
            "class": "GPT4V",
            "model": "gpt-4o-2024-08-06",
            "temperature": 0,
            "img_detail": "high"
        },
        "GPT4o_20240806_T10_Low": {
            "class": "GPT4V",
            "model": "gpt-4o-2024-08-06",
            "temperature": 1.0,
            "img_detail": "low"
        },
        "GPT4o_20241120": {}
    },
    "data": {
        "MME-RealWorld-Lite": {
            "class": "MMERealWorld",
            "dataset": "MME-RealWorld-Lite"
        },
        "MMBench_DEV_EN_V11": {
            "class": "ImageMCQDataset",
            "dataset": "MMBench_DEV_EN_V11"
        },
        "MMBench_Video_8frame_nopack":{},
        "Video-MME_16frame_subs": {
            "class": "VideoMME",
            "dataset": "Video-MME",
            "nframe": 16,
            "use_subtitle": true
        }
    }
}
```

配置json的解释：

1. 现在我们支持两个字段：`model`和`data`，每个字段都是一个字典。字典的键是模型/数据集的名称（由用户设置），值是模型/数据集的设置。
2. 对于`model`中的项目，值是一个包含以下键的字典：
   - `class`：模型的类名，应该是`vlmeval/vlm/__init__.py`（开源模型）或`vlmeval/api/__init__.py`（API模型）中定义的类名。
   - 其他kwargs：其他kwargs是模型特定的参数，请参考模型类的定义以获取详细用法。例如，`model`、`temperature`、`img_detail`是`GPT4V`类的参数。值得注意的是，大多数模型类都需要`model`参数。
   - Tip：在位于`vlmeval/config.py`的变量`supported_VLM`中的已经被定义的模型可以作为`model`的键，而不需要填对应的值即可启动。例如，`GPT4o_20240806_T00_HIGH: {}`是等价于`GPT4o_20240806_T00_HIGH: {'class': 'GPT4V', 'model': 'gpt-4o-2024-08-06', 'temperature': 0, 'img_size': -1, 'img_detail': 'high', 'retry': 10, 'verbose': False}`。
3. 对于字典`data`，我们建议用户使用官方数据集名称作为键（或键的一部分），因为我们经常根据数据集名称确定后处理/判断设置。对`data`中的项目，值是一个包含以下键的字典：
   - `class`：数据集的类名，应该是`vlmeval/dataset/__init__.py`中定义的类名。
   - 其他kwargs：其他kwargs是数据集特定的参数，请参考数据集类的定义以获取详细用法。通常，大多数数据集类都需要`dataset`参数。大多数视频数据集类都需要 `nframe` 或 `fps` 参数。
   - Tip：在位于`vlmeval/dataset/video_dataset_config.py`的变量`supported_video_dataset`中的已经被定义的数据集可以作为`data`的键，而不需要填对应的值即可启动。例如，`MMBench_Video_8frame_nopack: {}`是等价于`MMBench_Video_8frame_nopack: {'class': 'MMBenchVideo', 'dataset': 'MMBench-Video', 'nframe': 8, 'pack': False}`。

命令：

```python
python run.py --config config.json
```

## 2.3 好用的命令

```python
pkill -f python
pkill -f vllm
nvidia-smi
```

## 2.4 Prompt

在评估Qwen3-VL-4B-Instruct模型时要注意修改build_prompt函数中的prompt，让基座模型可以正确输出我们想要的格式。值得注意的是，不同的prompt对模型的正确率有影响，最好保持训测一致。

```python
prompt = f"*****************,such as{}."
```

## 2.5 如何让模型输出指定的json格式

- prompt：明确要求，比如“请严格按照以下格式输出，不要添加任何markdown标记或额外文字”，然后给出一个完整的例子
- constrained decoding：在生成的时候限制词表，只让模型输出符合json语法的token。
- function call：定义一个json schema，模型会按照格式生成结构化输出。缺点是不是所有模型都支持。
- 后处理：去掉不规范的标记，如果没有total_count则计算输出框or点的个数。
- temperature设置低一点。
- 重试机制+验证严格

# 3.COCO数据集

## 3.1 数据集质量

在训练过程中，存在一些标注错误的图片，比如在coco/annotations/instances_train2017.json中的000000507318.jpg就把菠萝错标注成了香蕉

# 4.Qwen3-VL的计数

## 4.1 Box-based Grounding

COCO:

Objects365:

OpenImages:

RefCOCO：RefCOCO 数据集是一个专门为引用表达生成（REG）任务设计的数据集，它旨在帮助研究者更好地理解如何在自然语言表达中指向图像中的特定对象。**视觉定位数据集**

## 4.2 Point-based Grounding

（i）来自PixMo的公共指向和计数注释（Deitke等人，2024）;

（ii）从公共对象检测和实例分割基准导出的对象基础数据;

（iii）由专用合成流水线生成的高精度指向注释

PixMo：PixMo是一个用于预训练和微调视觉语言模型的数据集集合，包含多个子数据集，如PixMo-Cap（用于预训练和微调的高度详细的密集标题数据集）、PixMo-AskModelAnything（包含人类编写的图像-问题-答案三元组的指令微调数据）等。所有数据集均未使用视觉语言模型构建。

## 4.3 Counting

高质量子集

# 5. 经验总结（计数问题）

1.多种prompt比单一prompt训练更稳定

2.多类和单类训练效果差不多

3.对于计数问题，特别是直接计数问题回答的格式要设计的十分简洁

# 6. Megatron

## 6.1 关于Megatron中--train-iters和--max_epochs两个参数介绍

[Megatron中--train-iters和--max_epochs两个参数介绍_ms-swift megatron-CSDN博客](https://blog.csdn.net/shizheng_Li/article/details/157658584)

## 6.2 megatron和hugging face模型区别

[(4 封私信 / 80 条消息) Hugging Face 和Megatron模型格式详解 - 以 Qwen3-0.6B 为例 - 知乎](https://zhuanlan.zhihu.com/p/1986585747843724685)

# 7. 工具包

## 计算tokens

```python
import json
from transformers import AutoTokenizer

# 原始字符串
json_str = """{
        "system": "银河证券APP系统",
        "module": "财富星Plus分布式",
        "feature": "修改收费方式",
        "intent": "验证从'全账户提佣'修改为'单笔交易固定收费'",
        "pre_condition": "用户已登录系统，并已签约'全账户提佣'收费方式",
        "test_steps": "1. 进入财富星Plus分布式模块\\n2. 选择已签约的'全账户提佣'收费方式\\n3. 点击'修改收费方式'按钮\\n4. 选择'单笔交易固定收费'\\n5. 确认修改",
        "expect_result": "收费方式成功修改为'单笔交易固定收费'",
        "is_negative": "正例",
        "rules": "将已经签约的收费方式改为另外一种收费方式。",
        "node_id": "CHARGE_006"
    },"""

# 使用Qwen的tokenizer进行精确计算
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-72B")
tokens = tokenizer.tokenize(json_str)
token_count = len(tokens)

print(f"精确Token计数: {token_count}")
```

# 8.问题

## 为什么模型在视觉计数方面表现那么差？

在图像计数任务中，大模型面临的主要挑战包括：一是视觉识别的局限性，如无法准确检测重叠物体；二是常识先验的干扰，当视觉模块检测到非常规数量时，语言模块可能因常识先验强制修正答案（如将六指修正为五指）；三是注意力机制的计算复杂度，处理长序列或复杂图像时，计算复杂度呈二次型增长，导致性能下降 。
