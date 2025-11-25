### V1.0 合成世界QA与流水线联调
- 时间：1125 10:40
python offline/run_pipeline.py --config configs/synth_world.yaml --steps encode index --force
th_world_llm.jsonl D:\Code\kblam_pp\kblam_pp\store\synth_world_llm --d_k 384 --d_v 384 --d_ctx 384 --d_tau 32 --model BAAI/bge-small-en-v1.5 --batch_size 64 --max_length 128
Encoded 80 five-tuples into D:\Code\kblam_pp\kblam_pp\store\synth_world_llm
[pipeline] python D:\Code\kblam_pp\kblam_pp\offline\build_index.py --store_dir D:\Code\kblam_pp\kblam_pp\store\synth_world_llm --method hnsw
Loaded keys of shape (80, 384)
Index saved to D:\Code\kblam_pp\kblam_pp\store\synth_world_llm\index_hnsw
[pipeline] Completed requested steps
(kblampp) PS D:\Code\kblam_pp\kblam_pp> python -m train.train_stageA --config configs/synth_world.yaml --max_steps 50
Traceback (most recent call last):
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\utils\_http.py", line 402, in hf_raise_for_status
    response.raise_for_status()
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\requests\models.py", line 1026, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\transformers\utils\hub.py", line 479, in cached_files
    hf_hub_download(
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\utils\_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\file_download.py", line 1007, in hf_hub_download      
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\file_download.py", line 1114, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\file_download.py", line 1655, in _raise_on_head_call_error
    raise head_call_error
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\file_download.py", line 1543, in _get_metadata_or_catch_error
    metadata = get_hf_file_metadata(
               ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\utils\_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\file_download.py", line 1460, in get_hf_file_metadata 
    r = _request_wrapper(
        ^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\file_download.py", line 283, in _request_wrapper      
    response = _request_wrapper(
               ^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\file_download.py", line 307, in _request_wrapper      
    hf_raise_for_status(response)
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\huggingface_hub\utils\_http.py", line 419, in hf_raise_for_status     
    raise _format(GatedRepoError, message, response) from e
huggingface_hub.errors.GatedRepoError: 401 Client Error. (Request ID: Root=1-692511d1-7563a73027bdd15c4d21e206;731d9754-2fa9-40c8-99de-ea64d10b6eec)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json.
Access to model meta-llama/Llama-3.2-1B-Instruct is restricted. You must have access to it and be authenticated to access it. Please log in.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "D:\Code\kblam_pp\kblam_pp\train\train_stageA.py", line 154, in <module>
    main()
  File "D:\Code\kblam_pp\kblam_pp\train\train_stageA.py", line 74, in main
    backbone = AutoModelForCausalLM.from_pretrained(model_name).to(device)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\transformers\models\auto\auto_factory.py", line 549, in from_pretrained
    config, kwargs = AutoConfig.from_pretrained(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\transformers\models\auto\configuration_auto.py", line 1332, in from_pretrained
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\transformers\configuration_utils.py", line 662, in get_config_dict    
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\transformers\configuration_utils.py", line 721, in _get_config_dict   
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\transformers\utils\hub.py", line 322, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\transformers\utils\hub.py", line 543, in cached_files
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       
  File "C:\Users\wanf1\miniconda3\envs\kblampp\Lib\site-packages\transformers\utils\hub.py", line 543, in cached_files
    raise OSError(
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct.     
401 Client Error. (Request ID: Root=1-692511d1-7563a73027bdd15c4d21e206;731d9754-2fa9-40c8-99de-ea64d10b6eec)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json.
Access to model meta-llama/Llama-3.2-1B-Instruct is restricted. You must have access to it and be authenticated to access it. Please log in.