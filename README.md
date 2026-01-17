# My LangChain Learning Project

LangChain 學習專案，涵蓋從基礎 LLM 使用到進階 Multi-Agent 系統的完整學習路徑。

參考資源：https://github.com/langchain-ai/learning-langchain

---

## 目錄

- [環境設定](#環境設定)
- [Chapter 1: LLM 基礎與 Prompt 工程](#chapter-1-llm-基礎與-prompt-工程)
- [Chapter 2: 文件載入與向量嵌入](#chapter-2-文件載入與向量嵌入)
- [Chapter 3: RAG 檢索增強生成](#chapter-3-rag-檢索增強生成)
- [Chapter 4: 記憶管理](#chapter-4-記憶管理)
- [Chapter 5: 進階應用](#chapter-5-進階應用)
- [Chapter 6: Agent 設計模式](#chapter-6-agent-設計模式)
- [Chapter 7: 反思與多代理系統](#chapter-7-反思與多代理系統)
- [Chapter 8: 進階功能](#chapter-8-進階功能)

---

## 環境設定

```bash
# Python 版本
Python 3.11+

# 安裝依賴
pip install langchain langchain-openai langchain-community langchain-postgres
pip install langgraph beautifulsoup4 pypdf psycopg

# 環境變數 (.env)
OPENAI_API_KEY=your_api_key
```

---

## Chapter 1: LLM 基礎與 Prompt 工程

本章介紹 LangChain 的核心概念，從基本的 LLM 呼叫到進階的組合模式。

| 檔案 | 技術重點 |
|------|----------|
| `a-llm.py` | **ChatOpenAI 基礎使用** - 使用 `ChatOpenAI` 模型直接傳入字串作為提示詞，回應物件包含 `content`, `additional_kwargs`, `response_metadata`, `usage_metadata` |
| `b-chat.py` | **HumanMessage 訊息格式** - 使用 `HumanMessage` 類別封裝使用者訊息，是 LangChain 標準的訊息格式 |
| `c-system.py` | **SystemMessage 系統角色** - 結合 `SystemMessage` 和 `HumanMessage` 設定 AI 助手的行為角色 |
| `d-prompt.py` | **PromptTemplate 提示詞範本** - 使用 `PromptTemplate.from_template()` 建立可重複使用的提示詞範本，支援變數插入 (`{context}`, `{question}`) |
| `e-prompt-model.py` | **範本與模型串接** - 將 `PromptTemplate` 產生的提示詞傳入 `ChatOpenAI` 模型執行 |
| `f-chat-promptl.py` | **ChatPromptTemplate 對話範本** - 使用 `ChatPromptTemplate.from_messages()` 建立多角色對話範本，適合對話任務 |
| `g-chat-prompt-model.py` | **完整對話流程** - 整合 `ChatPromptTemplate` 與 `ChatOpenAI`，建立完整的問答流程 |
| `h-structured.py` | **結構化輸出** - 使用 Pydantic `BaseModel` 定義輸出結構，透過 `with_structured_output()` 確保 LLM 回傳指定格式 |
| `i-csv.py` | **CommaSeparatedListOutputParser** - 使用輸出解析器將逗號分隔字串轉換為 Python list |
| `j-methods.py` | **三種呼叫方式** - `invoke()` 單次呼叫、`batch()` 批次呼叫、`stream()` 串流呼叫，並比較性能差異 |
| `k-imperative.py` | **命令式組合 (Imperative)** - 使用 `@chain` 裝飾器將函數轉換為 LangChain Runnable，實現自訂邏輯 |
| `ka-stream.py` | **命令式串流模式** - 結合 `@chain` 與 `yield` 實現邊產生邊回傳的串流效果 |
| `kn-async.py` | **非同步模式** - 使用 `async/await` 與 `ainvoke()` 實現非同步呼叫 |
| `l-declarative.py` | **宣告式組合 (Declarative)** - 使用 `\|` 運算子串接元件 (`template \| model`)，LangChain 推薦的組合方式 |

---

## Chapter 2: 文件載入與向量嵌入

本章涵蓋 RAG 系統的資料準備階段：文件載入、文字切割、向量嵌入與向量資料庫。

| 檔案 | 技術重點 |
|------|----------|
| `a.text-loader.py` | **TextLoader 文字檔載入** - 使用 `TextLoader` 讀取本地文字檔，支援指定編碼 |
| `b.web-loader.py` | **WebBaseLoader 網頁載入** - 使用 `WebBaseLoader` 抓取網頁內容 (需安裝 beautifulsoup4) |
| `c.pdf-loader.py` | **PyPDFLoader PDF 載入** - 使用 `PyPDFLoader` 讀取 PDF 檔案 (需安裝 pypdf) |
| `d.rec-text-splitter.py` | **RecursiveCharacterTextSplitter** - 遞迴字元切割器，設定 `chunk_size` (區塊大小) 和 `chunk_overlap` (重疊字元數) |
| `e.rec-text-splitter-code.py` | **程式碼切割** - 使用 `RecursiveCharacterTextSplitter.from_language()` 按程式語言邏輯分割 (支援 Python, JS 等) |
| `f.markdown-splitter.py` | **Markdown 切割** - 針對 Markdown 格式切割，可附加 `metadata` 記錄來源資訊 |
| `g.embeddings.py` | **OpenAIEmbeddings 向量嵌入** - 使用 `text-embedding-3-small` 模型將文字轉換為向量，支援自訂維度 |
| `h.load-split-embed.py` | **完整流程** - 整合載入 → 切割 → 嵌入的完整資料處理流程 |
| `i.pg-vector.py` | **PGVector 向量資料庫** - 使用 PostgreSQL + pgvector 儲存向量，支援 `similarity_search()` 相似度搜尋、新增/刪除文件 |
| `j.record-manager.py` | **SQLRecordManager 增量索引** - 使用 Record Manager 追蹤文件變更，支援 `incremental` 模式避免重複寫入，自動處理更新與刪除 |

### PGVector Docker 設定
```bash
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
```

---

## Chapter 3: RAG 檢索增強生成

本章深入 RAG 技術，包含多種查詢優化策略。

| 檔案 | 技術重點 |
|------|----------|
| `a-basic-rag.py` | **基礎 RAG** - 建立檢索器 (`as_retriever()`)，查詢相關文件後結合 LLM 產生答案。Query 使用兩次：一次 RAG 搜尋、一次 LLM 生成 |
| `b-rewrite.py` | **Query Rewrite (RRR)** - 重寫使用者查詢以提高檢索準確度，移除無關資訊，產生更精確的搜尋查詢 |
| `c-multi-query.py` | **Multi-Query** - 將原始問題產生 5 個不同角度的查詢，克服距離相似度搜尋的限制，使用 `get_unique_union()` 合併去重 |
| `d-rag-fusion.py` | **RAG Fusion + RRF** - 產生多個查詢後使用 Reciprocal Rank Fusion (RRF) 演算法重新排序結果。RRF 公式：`1/(rank + k)`，跨查詢累加分數 |
| `e-hyde.py` | **HyDE (Hypothetical Document Embeddings)** - 先讓 LLM 產生假設性回答，再用該回答進行 RAG 檢索，提高相關性匹配 |
| `f-router.py` | **Logical Router** - 使用 `with_structured_output()` 搭配 Pydantic 模型實現查詢路由，根據問題類型選擇不同資料來源 |
| `g-semantic-router.py` | **Semantic Router** - 使用向量相似度 (Cosine Similarity) 比較查詢與預設 Prompt 範本，自動選擇最適合的處理流程 |

---

## Chapter 4: 記憶管理

本章介紹 LangGraph 的狀態管理與對話記憶機制。

| 檔案 | 技術重點 |
|------|----------|
| `a-simple-memory.py` | **簡單記憶** - 使用 `placeholder` 在 `ChatPromptTemplate` 中傳入歷史訊息，手動管理對話歷史 |
| `b-state-graph.py` | **StateGraph 基礎** - 使用 `StateGraph` 建立工作流，定義 `State` (狀態)、`Node` (節點函數)、`Edge` (連結)。`Annotated[list, add_messages]` 自動累積訊息 |
| `c-persistent-memory.py` | **持久化記憶** - 使用 `MemorySaver` 作為 checkpointer，透過 `thread_id` 區分不同對話，支援跨呼叫記憶 |
| `d-trim-message.py` | **訊息修剪 trim_messages** - 根據 token 數量限制修剪訊息，支援 `strategy="last"` (保留最新)、`include_system=True` (保留系統訊息)、`start_on="human"` (從人類訊息開始) |
| `e-filter-message.py` | **訊息過濾 filter_messages** - 根據類型、名稱、ID 過濾訊息，如 `include_types="human"`、`exclude_names=[...]` |
| `f-merge-message.py` | **訊息合併 merge_message_runs** - 合併連續相同類型的訊息，減少訊息數量 |

---

## Chapter 5: 進階應用

本章展示完整的 LangGraph 應用案例。

| 檔案 | 技術重點 |
|------|----------|
| `a-chatbot.py` | **基礎 Chatbot** - 使用 StateGraph 建立簡單的聊天機器人，示範 `stream()` 串流輸出 |
| `b-sql-generator.py` | **SQL 生成器** - 多節點工作流：`generate_sql` → `explain_sql`。使用不同溫度模型 (0.1 生成 SQL, 0.7 解釋)，定義 `Input`/`Output` Schema 限制輸入輸出 |
| `c-multi-rag.py` | **多領域 RAG** - 使用 `router_node` 判斷查詢領域 (醫療記錄/保險)，`add_conditional_edges()` 實現動態路由到不同檢索器，`InMemoryVectorStore` 作為輕量向量儲存 |

---

## Chapter 6: Agent 設計模式

本章介紹 LangGraph Agent 的核心概念：思維鏈與工具使用。

| 檔案 | 技術重點 |
|------|----------|
| `a-basic-agent.py` | **基礎 Agent** - 使用 `@tool` 裝飾器定義工具 (calculator, DuckDuckGoSearch)，`bind_tools()` 綁定工具到模型，`ToolNode` + `tools_condition` 建立工具呼叫迴圈 |
| `b-force-first-tool.py` | **強制首次工具呼叫** - 新增 `first_model` 節點，強制第一步執行搜尋工具，使用 `ToolCall` 手動建立工具呼叫 |
| `c-main-tools.py` | **動態工具選擇** - 使用向量相似度從大量工具中選擇最相關的工具，`select_tools` 節點根據查詢動態綁定工具到模型 |

### Agent 架構圖
```
START → model → tools_condition
                    ↓
            tools ←→ model → END
```

---

## Chapter 7: 反思與多代理系統

本章涵蓋 Agent 自我改進與多 Agent 協作模式。

| 檔案 | 技術重點 |
|------|----------|
| `a-reflection.py` | **自我反思 (Reflection)** - `generate` 節點產生文章，`reflect` 節點評估並提供改進建議，透過 `should_continue()` 控制迭代次數，訊息角色互換實現自我對話 |
| `b-subgraph-direct.py` | **子圖直接嵌入** - 使用共享 key (`foo`) 直接將子圖作為節點嵌入父圖，狀態自動傳遞 |
| `c-subgraph-function.py` | **子圖函數封裝** - 父子圖使用不同 State Schema，在節點函數中手動轉換 key 名稱 (`foo` ↔ `bar`)，實現狀態隔離 |
| `d-supervisor.py` | **Supervisor 多代理模式** - Supervisor 節點使用 `with_structured_output()` 決定下一步 (`researcher`/`coder`/`FINISH`)，`MessagesState` 簡化狀態定義，`conditional_edges` 實現動態路由 |

### Supervisor 架構
```
START → supervisor → researcher → supervisor
                  ↘ coder     ↗
                     ↘ FINISH → END
```

---

## Chapter 8: 進階功能

本章介紹人機互動、授權控制與狀態編輯。

| 檔案 | 技術重點 |
|------|----------|
| `a-structured-output.py` | **結構化輸出** - 使用 Pydantic `BaseModel` + `Field(description=...)` 定義輸出格式，確保 LLM 回傳可解析的結構 |
| `b-streaming-output.py` | **串流輸出模式** - `stream_mode="updates"` 只輸出新增內容 vs `stream_mode="values"` 輸出完整狀態 |
| `c-interrupt.py` | **中斷點控制** - `interrupt_before=["chatbot", "tools"]` 在指定節點前暫停，`graph.get_state()` 檢查 `state.next` 判斷是否有待執行節點 |
| `d-authorize.py` | **工具授權** - 在 `tools` 節點前中斷，檢查 `tool_calls` 內容，讓使用者決定是否授權執行敏感操作 |
| `e-edit-state.py` | **狀態編輯** - `graph.update_state()` 動態修改狀態，`RemoveMessage` 清除訊息避免 tool_calls 殘留錯誤，`as_node="__start__"` 重設流程起點 |

### 人機互動流程
```
執行 → 中斷 → 顯示待執行操作 → 使用者授權 (y/n/e)
                                    ↓
                           [y] 繼續執行
                           [n] 結束
                           [e] 編輯 prompt，重新執行
```

---

## 技術總結

| 類別 | 核心技術 |
|------|----------|
| **LLM 呼叫** | ChatOpenAI, invoke/batch/stream, 結構化輸出 |
| **Prompt 工程** | PromptTemplate, ChatPromptTemplate, 變數插入 |
| **文件處理** | TextLoader, WebBaseLoader, PyPDFLoader, RecursiveCharacterTextSplitter |
| **向量嵌入** | OpenAIEmbeddings, PGVector, InMemoryVectorStore |
| **RAG 優化** | Query Rewrite, Multi-Query, RAG Fusion (RRF), HyDE, Routing |
| **狀態管理** | StateGraph, add_messages, MemorySaver, trim/filter/merge_messages |
| **Agent 設計** | @tool, bind_tools, ToolNode, tools_condition |
| **多代理系統** | Subgraph, Supervisor Pattern, 狀態隔離與轉換 |
| **人機互動** | interrupt_before, update_state, 授權控制 |

---

## 授權

MIT License
