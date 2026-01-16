from langchain_openai.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv(override=True)  
api_key = os.getenv("OPENAI_API_KEY")   

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
#三種呼叫方式：單次、批次、串流

# 單次呼叫
complete = model.invoke("Hi there!")
print(f"complete: {complete.content}")

# 批次呼叫
# completes = model.batch(["Hi there!", "Bye now!"])
# print(f"completes: {[c.content for c in completes]}")

# 串流呼叫
# for token in model.stream("bye!"):
#     print(token.content)

# =====================================
# 性能比較：invoke vs batch
# =====================================
# prompts = ["Hi there!", "Bye now!"]

# # 方法1: 呼叫兩次 invoke
# print("=" * 50)
# print("方法1: 呼叫兩次 invoke")
# start_time = time.time()
# results_invoke = [model.invoke(prompt) for prompt in prompts]
# invoke_time = time.time() - start_time
# print(f"耗時: {invoke_time:.4f} 秒")
# print(f"結果: {[r.content for r in results_invoke]}")

# # 方法2: 呼叫一次 batch
# print("\n" + "=" * 50)
# print("方法2: 呼叫一次 batch")
# start_time = time.time()
# results_batch = model.batch(prompts)
# batch_time = time.time() - start_time
# print(f"耗時: {batch_time:.4f} 秒")
# print(f"結果: {[r.content for r in results_batch]}")

# # 比較結果
# print("\n" + "=" * 50)
# print("性能比較結果:")
# print(f"invoke 耗時: {invoke_time:.4f} 秒")
# print(f"batch 耗時:  {batch_time:.4f} 秒")
# if invoke_time > batch_time:
#     improvement = ((invoke_time - batch_time) / invoke_time) * 100
#     print(f"✓ batch 更快，快 {improvement:.2f}%")
# else:
#     improvement = ((batch_time - invoke_time) / batch_time) * 100
#     print(f"✓ invoke 更快，快 {improvement:.2f}%")