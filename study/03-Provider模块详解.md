# Provider 模块：LLM API 调用全链路

> 追踪一次 `Stream()` 调用从构造请求、发送 HTTP、接收 SSE 流、到解析为 Chunk 的完整过程。
> 覆盖 OpenAI 和 Anthropic 两个 provider，以及重试、缓存命中等机制。

---

## 一、模块结构

```
internal/provider/
├── provider.go                ← 核心类型定义（Message, Request, Chunk, Provider 接口, Usage, Pricing）
├── retry.go                   ← HTTP 重试 + 指数退避 + 状态码判断
├── schema_canonicalize.go     ← JSON Schema 规范化（保证跨 turn 字节稳定）
├── pairing_probe_test.go      ← 工具调用配对修复的边界测试
├── openai/
│   ├── openai.go              ← OpenAI 兼容 provider（DeepSeek/MiMo 等）
│   ├── think.go               ← 思考标签 <think> 拆分器
│   └── fetch_models.go        ← 获取可用模型列表
└── anthropic/
    └── anthropic.go           ← Anthropic Messages API provider
```

---

## 二、核心类型定义（provider.go）

### Provider 接口

```go
// provider.go:220
type Provider interface {
    Name() string
    Stream(ctx context.Context, req Request) (<-chan Chunk, error)
}
```

极简——只有一个 `Stream` 方法。所有 provider 做的事就是：接收 `Request`，返回 `<-chan Chunk`。

### Request（请求入参）

```go
// provider.go:56
type Request struct {
    Messages    []Message     // 完整对话历史
    Tools       []ToolSchema  // 工具定义列表
    Temperature float64
    MaxTokens   int
}
```

### Chunk（流式出参）

```go
// provider.go:209
type Chunk struct {
    Type      ChunkType
    Text      string    // ChunkText, ChunkReasoning
    Signature string    // 推理签名（Anthropic）
    ToolCall  *ToolCall // 工具调用
    Usage     *Usage    // token 用量
    Err       error     // 错误
}
```

ChunkType 枚举：
- `ChunkText` — 普通文本增量
- `ChunkReasoning` — 思考/推理增量
- `ChunkToolCallStart` — 工具调用开始（只有 ID + Name）
- `ChunkToolCall` — 完整工具调用（参数已收齐）
- `ChunkUsage` — token 用量统计
- `ChunkDone` — 流正常结束
- `ChunkError` — 错误

### Usage（用量统计）

```go
// provider.go:172
type Usage struct {
    PromptTokens     int
    CompletionTokens int
    TotalTokens      int
    CacheHitTokens   int    // ← prompt 中命中缓存的 token 数
    CacheMissTokens  int    // ← prompt 中未命中缓存的 token 数
    ReasoningTokens  int    // ← 思考 token 子集
    FinishReason     string // ← "stop" | "tool_calls" | "length" | ...
}
```

---

## 三、Provider 注册机制

每个 provider 通过 `init()` 自注册：

```go
// openai/openai.go:22
func init() { provider.Register("openai", New) }

// anthropic/anthropic.go:45
func init() { provider.Register("anthropic", New) }
```

注册表是全局 map：

```go
// provider.go:261
var registry = map[string]Factory{}

func Register(kind string, f Factory) { ... }
func New(kind string, cfg Config) (Provider, error) { ... }
```

启动时通过 blank import 触发注册：

```go
// cmd/reasonix/main.go
import (
    _ "reasonix/internal/provider/anthropic"
    _ "reasonix/internal/provider/openai"
)
```

---

## 四、一次 Stream 调用的完整链路

### 4.1 顶层：`client.Stream()`（以 OpenAI 为例）

```go
// openai/openai.go:111
func (c *client) Stream(ctx context.Context, req provider.Request) (<-chan provider.Chunk, error) {
    // ① 构造请求体
    body, _ := json.Marshal(c.buildRequest(req))

    // ② 内联定义 HTTP 请求构造函数（重试时复用）
    newReq := func(ctx context.Context) (*http.Request, error) {
        httpReq, _ := http.NewRequestWithContext(ctx, "POST",
            c.baseURL+"/chat/completions", bytes.NewReader(body))
        httpReq.Header.Set("Content-Type", "application/json")
        httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
        httpReq.Header.Set("Accept", "text/event-stream")
        return httpReq, nil
    }

    // ③ 发送请求（带重试）
    resp, err := provider.SendWithRetry(ctx, c.http, c.name, c.keyEnv, newReq)

    // ④ 启动 goroutine 解析 SSE 流
    out := make(chan provider.Chunk)
    go c.readStream(ctx, resp, out)
    return out, nil
}
```

**关键时序：**
```
Stream() 调用
    │
    ├── buildRequest(req)          ← 同步：Message 转换 + Tools 序列化
    ├── SendWithRetry(...)         ← 同步阻塞：HTTP POST + 拿到 200 响应头
    ├── make(chan Chunk)           ← 创建 channel
    ├── go readStream(resp, out)   ← 异步 goroutine：读 body、解析 SSE
    └── return out, nil            ← 立即返回 channel
                                      │
调用方:  for chunk := range ch { ... }  ← 遍历 channel，边收边消费
```

---

### 4.2 `buildRequest()` — 把内部类型转成 API JSON

**OpenAI 兼容 API 版：**

```go
// openai/openai.go:137
func (c *client) buildRequest(req provider.Request) chatRequest {
    // ① 修复工具调用配对（中断恢复的会话可能有悬空调用）
    src := provider.SanitizeToolPairing(req.Messages)

    // ② 转换 messages：去掉 reasoning_content（不发给 LLM，省 token）
    for i, m := range src {
        cm := chatMessage{
            Role:    string(m.Role),
            Content: m.Content,
        }
        // 转换 tool_calls 结构
        for _, tc := range m.ToolCalls {
            wire := chatToolCall{ID: tc.ID, Type: "function"}
            wire.Function.Name = tc.Name
            wire.Function.Arguments = tc.Arguments
            cm.ToolCalls = append(cm.ToolCalls, wire)
        }
        msgs[i] = cm
    }

    // ③ 转换 tools
    for _, t := range req.Tools {
        tools = append(tools, chatTool{
            Type:     "function",
            Function: chatFunction{
                Name:        t.Name,
                Description: t.Description,
                Parameters:  t.Parameters,   // ← 已规范化的 JSON Schema
            },
        })
    }

    // ④ 组装最终请求
    return chatRequest{
        Model:           c.model,
        Messages:        msgs,
        Tools:           tools,
        Stream:          true,                     // ← 硬编码 true
        StreamOptions:   &streamOptions{IncludeUsage: true},
        Temperature:     req.Temperature,
        Thinking:        &thinkingMode{Type: "enabled"},  // DeepSeek 专用
    }
}
```

**Anthropic 版差异：**

```go
// anthropic/anthropic.go:131
func (c *client) buildRequest(req provider.Request) anthRequest {
    // ① system 消息提取到顶层 system 字段
    // ② tool 结果变成 user turn 中的 tool_result 块
    // ③ 连续同 role 的消息合并（Anthropic 强制要求 user/assistant 交替）
    // ④ 如果有签名 reasoning，作为 thinking 块回放（Anthropic 要求）
}
```

---

### 4.3 `readStream()` — SSE 流解析

这是两个 provider 差异最大的部分，因为 **SSE 事件格式完全不同**。

#### OpenAI 兼容版（openai/openai.go:193）

解析 OpenAI 标准的 SSE 流：
```
data: {"choices":[{"delta":{"content":"重"}}]}
data: {"choices":[{"delta":{"content":"构"}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read_file"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]}}]}
```

核心逻辑：**按 index 聚拢工具调用片段**：

```go
acc := map[int]*provider.ToolCall{}           // index → 工具调用状态
started := map[int]bool{}                     // 是否已发射 ChunkToolCallStart

for scanner.Scan() {
    line := scanner.Text()
    data := strings.TrimPrefix(line, "data:")
    var sr streamResponse
    json.Unmarshal([]byte(data), &sr)

    delta := sr.Choices[0].Delta

    // 文本 → 实时发射
    if delta.Content != "" { out <- Chunk{Type: ChunkText, Text: delta.Content} }

    // 推理 → 实时发射
    if delta.ReasoningContent != "" { out <- Chunk{Type: ChunkReasoning, ...} }

    // 工具调用 → 按 index 聚拢
    for _, tc := range delta.ToolCalls {
        cur := acc[tc.Index]
        cur.ID = tc.ID
        cur.Name = tc.Function.Name
        cur.Arguments += tc.Function.Arguments   // ← 拼接参数片段

        // 名字一出现就发射 start 事件（UI 早点展示卡片）
        if !started[tc.Index] && cur.Name != "" {
            out <- Chunk{Type: ChunkToolCallStart, ToolCall: cur}
        }
    }
}
// 流结束 → 按 index 顺序发射完整工具调用
for _, idx := range order {
    out <- Chunk{Type: ChunkToolCall, ToolCall: acc[idx]}
}
out <- Chunk{Type: ChunkDone}
```

#### Anthropic 版（anthropic/anthropic.go:241）

解析 Anthropic 的 `event:` + `data:` 格式：
```
event: content_block_start
data: {"type":"tool_use","id":"call_1","name":"read_file","index":0}

event: content_block_delta
data: {"type":"input_json_delta","partial_json":"{\"path\":"}

event: content_block_stop
data: {"index":0}
```

核心差异——**Anthropic 流里有明确的 start / delta / stop 事件类型**：

```go
switch ev.Type {
case "message_start":
    // 收集输入 token 用量（含缓存命中/未命中）
    cacheRead  = ev.Message.Usage.CacheReadInputTokens

case "content_block_start":
    if ev.ContentBlock.Type == "tool_use" {
        tc = &ToolCall{ID: ev.ContentBlock.ID, Name: ev.ContentBlock.Name}
        out <- Chunk{Type: ChunkToolCallStart, ToolCall: tc}
    }

case "content_block_delta":
    switch ev.Delta.Type {
    case "text_delta":       → ChunkText
    case "thinking_delta":   → ChunkReasoning
    case "signature_delta":  → ChunkReasoning(带签名)
    case "input_json_delta": → tc.Arguments += ev.Delta.PartialJSON
    }

case "content_block_stop":
    out <- Chunk{Type: ChunkToolCall, ToolCall: tc}  // 工具调用完整

case "message_delta":
    outTok = ev.Usage.OutputTokens

case "error":
    out <- ChunkError
}
```

---

## 五、重试机制（retry.go）

### SendWithRetry

```go
// retry.go:113
func SendWithRetry(ctx context.Context, httpClient *http.Client,
    provName, keyEnv string,
    newReq func(context.Context) (*http.Request, error),
) (*http.Response, error) {

    for attempt := 0; attempt <= MaxRetries; attempt++ {   // ← 最多 11 次（0+10）
        // backoff: 指数退避 + jitter + Retry-After 头
        delay := backoffDelay(attempt, retryAfter)
        time.Sleep(delay)

        resp, err := httpClient.Do(req)

        switch {
        case err != nil:
            // 网络错误 → 可重试（除了 context.Canceled / DeadlineExceeded）
            if transientErr(err) { lastErr = err; continue }

        case resp.StatusCode == 200:
            return resp, nil   // ← 成功

        case resp.StatusCode == 401 || 403:
            return &AuthError{...}  // ← 认证失败，不重试

        case RetryableStatus(resp.StatusCode):
            lastErr = apiErr; continue   // ← 408/429/5xx 可重试

        default:
            return apiErr  // ← 其他 4xx 不重试
        }
    }
    return nil, lastErr
}
```

### 重试策略

| 条件 | 行为 |
|------|------|
| 网络瞬时错误 | 重试（指数退避 + jitter） |
| HTTP 429（限流）| 重试，优先用 `Retry-After` 头 |
| HTTP 5xx | 重试 |
| HTTP 401/403 | **不重试**，直接返回 `AuthError` |
| HTTP 400/422 等 | **不重试**，客户端错误重试没用 |
| Context 取消 | **不重试**，调用方主动放弃 |

### 退避算法

```go
// retry.go:81
func backoffDelay(attempt int, retryAfter time.Duration) time.Duration {
    if retryAfter > 0 { return min(retryAfter, 15s) }  // 服务器说了算
    d := 500ms * 2^(attempt-1)                           // 指数：500ms → 1s → 2s → 4s → ...
    d += rand.Intn(250ms)                                 // 随机 jitter
    return min(d, 15s)                                    // 上限 15s
}
```

### 重试范围

**只重试到拿到 200 响应头为止。** SSE body 流式数据一旦开始就不重试——已经生成的 token 没法"退回"：

```go
// retry.go 注释：
// Retries cover only the header phase — once the body streams,
// mid-stream failures are not retried (the model has already emitted tokens).
```

---

## 六、缓存命中机制

### 缓存命中是什么？

DeepSeek 的 **自动前缀缓存**：如果连续两次请求的 **prompt 前缀**字节完全相同（system prompt + tools 列表），API 自动跳过这部分计算，token 不收费（或打折）。

### Reasonix 如何配合？

**1. 字节稳定保证缓存命中**

```go
// tool.go:196 — Schemas() 输出按名称排序 + 规范化
names := copy(r.order)
sort.Strings(names)   // ← 拼音序，杜绝抖动
for _, name := range names {
    out = append(out, ToolSchema{
        Parameters: r.canon[name],  // ← 已规范化的 JSON Schema（见下文）
    })
}
```

```go
// schema_canonicalize.go:10 — 递归稳定 JSON Schema
func CanonicalizeSchema(raw json.RawMessage) json.RawMessage {
    var v any
    json.Unmarshal(raw, &v)
    canonicalizeSchemaValue(v)     // ← 递归排序 required 数组、stabilize 字段
    return json.Marshal(v)
}
```

**2. Compose 里的设计**

Plan mode marker、记忆更新、后台任务结果都走 **turn tail**（塞进用户消息），不碰 system prompt 前缀，所以缓存不受影响。

**3. Agent 里的跟踪**

```go
// agent.go:140-141 — Agent 结构体中
sessCacheHit  atomic.Int64   // 本会话累计缓存命中 token 数
sessCacheMiss atomic.Int64   // 本会话累计缓存未命中 token 数

// agent.go:555-556 — stream() 中每轮更新
case ChunkUsage:
    a.sessCacheHit.Add(int64(chunk.Usage.CacheHitTokens))
    a.sessCacheMiss.Add(int64(chunk.Usage.CacheMissTokens))

// agent.go:381 — 发射给前端
a.sink.Emit(event.Event{Kind: event.Usage,
    SessionHit:  int(a.sessCacheHit.Load()),
    SessionMiss: int(a.sessCacheMiss.Load()),
})
```

**4. Provider 侧的归一化**

```go
// openai/openai.go:316 — normaliseUsage
// DeepSeek 用 prompt_cache_hit_tokens（顶层字段）
// OpenAI/MiMo 用 prompt_tokens_details.cached_tokens（嵌套字段）
// 统一归一化为 Usage.CacheHitTokens / CacheMissTokens

// anthropic/anthropic.go:342 — Anthropic 字段名又不一样
CacheHitTokens:  cacheRead,             // cache_read_input_tokens
CacheMissTokens: inTok + cacheCreate,    // 非缓存输入 + 缓存写入
```

**5. Pricing（计价）**

```go
// provider.go:191
func (p *Pricing) Cost(u *Usage) float64 {
    return (float64(u.CacheHitTokens)*p.CacheHit +    // 缓存命中 = 便宜
            float64(u.CacheMissTokens)*p.Input +       // 缓存未命中 = 全价
            float64(u.CompletionTokens)*p.Output) / 1e6
}
```

---

## 七、工具调用配对修复（SanitizeToolPairing）

会话中断恢复后的关键修复逻辑：

```go
// provider.go:78
func SanitizeToolPairing(msgs []Message) []Message {
    // assistant 带了 tool_calls →
    //     后面的 tool 消息按 ID 配对
    //     缺失的填充占位符 "[no result: ...]"
    //
    // 孤立的 tool 消息（前面没有 assistant tool_calls）→ 丢弃
}
```

API 要求"每个 tool_call 后面必须跟一个 tool_result"，不满足就 400。

---

## 八、两个 Provider 的全链路对比

```
OpenAI 兼容（DeepSeek / MiMo）               Anthropic（Claude）
─────────────────────────────────           ──────────────────────────
POST /chat/completions                      POST /v1/messages
Authorization: Bearer ...                   x-api-key: ...
                                            anthropic-version: 2023-06-01

{
  messages: [                               {
    {role:"system", content:"..."},           system: [{type:"text", text:"..."}],
    {role:"user",   content:"..."},           messages: [
    {role:"assistant", tool_calls:[...]},       {role:"user", content:[...]},
    {role:"tool", tool_call_id:"...",           {role:"assistant", content:[
      content:"..."}                                {type:"thinking", ...},  ← 签名 replay
  ]                                                 {type:"text", text:"..."},
  tools: [                                          {type:"tool_use", id:..., name:..., input:{}}]
    {type:"function", function:{...}}             ]}
  ]                                             }
                                              ]
SSE:                                         SSE:
data: {"choices":[{"delta":{...}}]}          event: content_block_start
data: {"choices":[{"delta":{...}}]}          data: {...}
                                             event: content_block_delta
                                             data: {...}
```

---

## 九、关键设计点总结

| 设计点 | 实现 |
|--------|------|
| **Provider 接口极简** | 只有一个 `Stream()` 方法返回 `<-chan Chunk` |
| **注册表模式** | `init()` + `provider.Register()`，编译时自动发现 |
| **流式硬编码** | `Stream: true` 永远开启，不做非流式 |
| **只重试头部** | body 开始流就不重试了 |
| **认证不重试** | 401/403 立刻返回 `AuthError`，含 actionable 消息 |
| **缓存稳定** | Schema 规范化、排序、turn tail 策略 |
| **工具配对修复** | `SanitizeToolPairing` 防中断恢复后的 400 |
| **多供应商归一化** | `normaliseUsage` 统一 DeepSeek/OpenAI/Anthropic 的用量格式 |
| **reasoning 不上传** | `reasoning_content` 不发给 LLM，省 500+ token/turn |
| **Anthropic 签名回放** | 有签名的 thinking block 必须回放，否则 API 拒绝 |
