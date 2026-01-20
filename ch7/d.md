shane@jichenxiangde-MacBook-Pro ch7 % uv run d-supervisor.py
/Users/shane/Documents/project_practice/My_LangChain/.venv/lib/python3.11/site-packages/langchain_openai/chat_models/base.py:2041: UserWarning: Cannot use method='json_schema' with model gpt-4 since it doesn't support OpenAI's Structured Output API. You can see supported models here: https://platform.openai.com/docs/guides/structured-outputs#supported-models. To fix this warning, set `method='function_calling'. Overriding to method='function_calling'.
warnings.warn(

========================
🧑‍💼 Supervisor evaluating state : {'messages': [HumanMessage(content='what is 4!', additional_kwargs={}, response_metadata={}, id='048627a1-2e0c-4e8f-8c46-92b2da7f96eb')], 'next': 'supervisor'}

========================
Step output:
{'supervisor': {'next': 'coder'}}

========================
Step output:
{'coder': {'messages': [AIMessage(content='The factorial of a non-negative integer n, denoted by n!, is the product of all positive integers less than or equal to n. \n\nSo, 4! is calculated as 4*3*2*1 = 24.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 49, 'prompt_tokens': 28, 'total_tokens': 77, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'id': 'chatcmpl-Czv0gAfP63uVzjqoHhwa6hxnUvhEy', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019bd90e-e2e8-78b3-bf19-2cc7c6377de8-0', usage_metadata={'input_tokens': 28, 'output_tokens': 49, 'total_tokens': 77, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}

========================
🧑‍💼 Supervisor evaluating state : {'messages': [HumanMessage(content='what is 4!', additional_kwargs={}, response_metadata={}, id='048627a1-2e0c-4e8f-8c46-92b2da7f96eb'), AIMessage(content='The factorial of a non-negative integer n, denoted by n!, is the product of all positive integers less than or equal to n. \n\nSo, 4! is calculated as 4*3*2*1 = 24.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 49, 'prompt_tokens': 28, 'total_tokens': 77, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'id': 'chatcmpl-Czv0gAfP63uVzjqoHhwa6hxnUvhEy', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019bd90e-e2e8-78b3-bf19-2cc7c6377de8-0', usage_metadata={'input_tokens': 28, 'output_tokens': 49, 'total_tokens': 77, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})], 'next': 'coder'}

========================
Step output:
{'supervisor': {'next': 'FINISH'}}
