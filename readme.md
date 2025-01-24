# E-LANG: Energy-Based Joint Inferencing of Super and Swift Language Models
This repo contains the demo and the implementation of the paper ['E-LANG: Energy-Based Joint Inferencing of Super and Swift Language Models'](https://aclanthology.org/2022.acl-long.359.pdf), published at ACL2022 (Oral).

Link to arXiv paper: https://arxiv.org/abs/2203.00748

![](https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/e-lang/elang_framework.png)

## Demo Video for T5-Based E-LANG
The following video demonstrates the performance of E-LANG with T5 backbone on a random subset of GLUE SST-2. T5-Large and T5-11B are respectively used as Swift and Super models in this demo.


```python
%%HTML
<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/e-lang/Demo.mp4" type="video/mp4">
</video>
```


<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/e-lang/Demo.mp4" type="video/mp4">
</video>



## Pytorch Implementation of E-LANG with BERT Backbone

#### Download Code and checkpoints

```python
!wget https://modelarts-cnnorth1-market-dataset.obs.cn-north-1.myhuaweicloud.com/example-apps/elang/code.zip
!unzip -qo code.zip
```

#### Requirements
In order to run the code, please install the required packages using the following command:


```python
!pip install -r requirements.txt
```

#### E-LANG over GLUE SST-2 with TinyBERT (as Swift) and BERT-Base (as Super):


```python
!python elang.py --do_eval --swift_model ./models/SST-2/TinyBERT_4L_312D/ --super_model ./models/SST-2/BERT_Base/ --router_threshold 1.31044 --task_name SST-2 --data_dir ./glue_data/SST-2/
```

    11/25 01:33:18 AM The args: Namespace(cache_dir='', data_dir='./glue_data/SST-2/', data_url='', do_eval=True, do_lower_case=False, eval_batch_size=1, eval_step=50, gradient_accumulation_steps=1, learning_rate=5e-05, max_seq_length=128, no_cuda=False, output_dir='outputs/', pred_distill=False, router_threshold=1.31044, seed=42, super_model='./models/SST-2/BERT_Base/', swift_model='./models/SST-2/TinyBERT_4L_312D/', task_name='SST-2', temperature=1.0, weight_decay=0.0001)
    11/25 01:33:18 AM device: cuda n_gpu: 1
    11/25 01:33:18 AM Writing example 0 of 872
    11/25 01:33:18 AM *** Example ***
    11/25 01:33:18 AM guid: dev-1
    11/25 01:33:18 AM tokens: [CLS] it ' s a charming and often affecting journey . [SEP]
    11/25 01:33:18 AM input_ids: 101 2009 1005 1055 1037 11951 1998 2411 12473 4990 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    11/25 01:33:18 AM input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    11/25 01:33:18 AM segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    11/25 01:33:18 AM label: 1
    11/25 01:33:18 AM label_id: 1
    11/25 01:33:18 AM Model config {
      "attention_probs_dropout_prob": 0.1,
      "finetuning_task": "sst-2",
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "num_labels": 2,
      "output_attentions": true,
      "output_hidden_states": true,
      "output_intermediate": true,
      "output_past": true,
      "pre_trained": "",
      "pruned_heads": {},
      "torchscript": false,
      "training": "",
      "type_vocab_size": 2,
      "use_bfloat16": false,
      "vocab_size": 30522
    }
    
    11/25 01:33:21 AM Loading model ./models/SST-2/BERT_Base/pytorch_model.bin
    11/25 01:33:22 AM loading model...
    11/25 01:33:22 AM done!
    11/25 01:33:22 AM Weights of TinyBertForSequenceClassification not initialized from pretrained model: ['fit_dense.weight', 'fit_dense.bias']
    11/25 01:33:25 AM Model config {
      "attention_probs_dropout_prob": 0.1,
      "gradient_checkpointing": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 312,
      "initializer_range": 0.02,
      "intermediate_size": 1200,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "model_type": "bert",
      "num_attention_heads": 12,
      "num_hidden_layers": 4,
      "output_intermediate": false,
      "pad_token_id": 0,
      "position_embedding_type": "absolute",
      "pre_trained": "",
      "training": "",
      "type_vocab_size": 2,
      "use_cache": true,
      "vocab_size": 30522
    }
    
    11/25 01:33:26 AM Loading model ./models/SST-2/TinyBERT_4L_312D/pytorch_model.bin
    11/25 01:33:26 AM loading model...
    11/25 01:33:26 AM done!
    11/25 01:33:26 AM ***** Running evaluation *****
    11/25 01:33:26 AM   Num examples = 872
    11/25 01:33:26 AM   Batch size = 1
    ********** threshold = 1.31044**************
    Evaluating: 100%|████████████████████████████| 872/872 [00:07<00:00, 113.55it/s]
    Samples Processed by Super (%): 23.62385321100917
    Samples Processed by Swift (%): 76.37614678899082
    Average Latency (s): 0.008390570179038092
    11/25 01:33:34 AM ***** Eval results *****
    11/25 01:33:34 AM   acc = 0.9369266055045872
    11/25 01:33:34 AM   eval_loss = 0.25824264600505725


#### Arguments
##### - 'do_eval': run the evaluation script
##### - 'swift_model': the checkpoint path for the Swift model
##### - 'super_model': the checkpoint path for the Super model
##### - 'router_threshold': the router threshold
##### - 'task_name': the GLUE task name (e.g., SST-2, QNLI, etc.)
##### - 'data_dir': the path to the dataset

#### E-LANG results with BERT on GLUE devset compared with SOTA

![](https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/e-lang/table.png)

#### Sample Trade-off curves compared with SOTA

![](https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/e-lang/curves.png)
