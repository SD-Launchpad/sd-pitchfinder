# PitchFinder

内部 CLI 工具——给定一段产品发布描述，从已知 AI/tech creator 库里挑出最值得 pitch 的人，附带他们最近相关内容、verified context、联系方式和个性化 pitch angle。

单用户工具。无 auth / 无 SaaS / 无多租户。

数据源：newsletter（Substack/Beehiiv/个人博客）、podcast、YouTube 频道——全走 RSS/Atom 统一抓取。

---

## 一、安装

需要 Python 3.11+（实测 3.12）。

```bash
cd shanda/pitchfinder
uv venv --python 3.12 .venv          # 或: python3 -m venv .venv
source .venv/bin/activate
uv pip install -e . --python .venv/bin/python    # 或: pip install -e .
```

填 API key——复制模板：

```bash
cp .env.example .env
```

然后编辑 `.env` 填两个 key：

| 变量 | 说明 |
|---|---|
| `OPENROUTER_API_KEY` | 用于 scoring + pitch angles + lint。必填。`sk-or-...` |
| `MIROMIND_API_KEY` | 用于 `deep-dive` 命令（top 5 creator 做 verified web search）。非必填——只跑前 8 步不需要。`sk_live_...` |

---

## 二、推荐 SOP（一次完整 PR campaign）

```bash
# 一次性
pitchfinder init                              # 建 SQLite schema (6 张表)
pitchfinder load seed_creators.yaml           # upsert 29 个 seed creator

# 每次启动 PR campaign 前
pitchfinder lint                              # 30 秒, ~$0.01, 验证所有 url 内容真实
pitchfinder refresh                           # 拉取所有 RSS feeds 最新内容(90天)

# 跑一次具体 launch
pitchfinder search "Your launch description here" \
  --min-score 55 --output reports/launch.html

# 深度增强 top 5（live web search + verified quotes + 联系方式）
pitchfinder deep-dive <search_id> --top 5

# 剩余 ranks 6-13 用 Sonnet 4.6 best-effort
pitchfinder deep-dive <search_id> --only <id1>,<id2>,... \
  --model anthropic/claude-sonnet-4.6

# 重渲染 HTML（合并 deep-dive 结果）
pitchfinder show <search_id> --min-score 55 \
  --output reports/launch-final.html

# 实施 pitch 时跟踪状态
pitchfinder status <creator_id> <campaign_name> pitched --notes "sent 2026-05-17"
```

---

## 三、模型搭配（按"scoring 便宜 + 报告准确可靠"）

| 阶段 | 推荐模型 | 调用量 | 阶段成本 | 选这个的理由 |
|---|---|---|---|---|
| **Scoring** (Relevance) | `deepseek/deepseek-chat-v3.1` | ~1600 | ~$0.15 | 0% fail 实测稳定；便宜；scoring 只是过滤器 |
| **Pitch angles** | `anthropic/claude-sonnet-4.6` | ~25 | ~$0.15 | 100% 稳；instruction-following 强；这是给你看的内容 |
| **Deep-dive top 5** | `mirothinker-1-7-deepresearch` | 5 | ~$2.5 | 真做 web search + verification，能找最新作品引用、当前 stance、联系方式 |
| **Deep-dive ranks 6-13** | `anthropic/claude-sonnet-4.6` | 8 | ~$0.20 | Best-effort 训练数据，honest 标注不确定 |
| **Lint (URL 内容验证)** | `deepseek/deepseek-chat-v3.1` | ~30 | ~$0.01 | Haiku-class 够用，量小不在乎模型品级 |
| **总计 / campaign** | | | **~$3** | |

实测过不稳的（OpenRouter 路由层 short-JSON 输出失败率高）：
- `deepseek/deepseek-v4-pro` (40% fail)
- `deepseek/deepseek-v4-flash` (偶发空响应)
- `z-ai/glm-5.1` (86% fail)

要换模型，改 `.env`：

```
PITCHFINDER_RELEVANCE_MODEL=deepseek/deepseek-chat-v3.1
PITCHFINDER_PITCH_MODEL=anthropic/claude-sonnet-4.6
MIROMIND_DEEPRESEARCH_MODEL=mirothinker-1-7-deepresearch
```

任何 [OpenRouter model id](https://openrouter.ai/models) 都能用。

---

## 四、命令详解

### `init`
建 SQLite schema（6 张表 + 索引）。当前 DB 在 `./pitchfinder.db`（gitignored）。

### `load <yaml>`
upsert seed creators。`ON CONFLICT(platform, handle) DO UPDATE`，重跑不会重复。

### `lint`
**内容感知**的 URL 验证器——比单纯 HTTP 200 检查更严。检测 5 种状态：
- `ok` — 真的是 creator 的频道
- `squatter` — GoDaddy/Sedo 域名待售页（HTTP 200 但实际是停泊）
- `wrong_content` — 页面是别人的
- `uncertain` — 反爬或 CAPTCHA 导致看不清
- `unreachable` — DNS 失败 / 4xx / 5xx / 超时

源于真实事故：`nopriors.com` HTTP 200 但实际是 GoDaddy 待售页，骗过了纯状态码检查。

跑一次 ~30 秒 + ~$0.01，每次 `load` 后建议跑。

### `refresh [--lookback-days 90] [--platforms substack,podcast,youtube]`
抓所有 creator 的 feed_url，新 item 入库。URL UNIQUE，重跑不重复。失败的 feed 不抛错只 log。

### `search "<description>" [--min-score 70] [--max-creators 30] [--output file.html|md]`

四步流水：
1. **Topic extraction** — Haiku-class 从描述提取 topics/keywords
2. **Relevance scoring** — 每个 item 在 lookback 窗口里被打 0-100 分，并发 8 worker
3. **Creator ranking** — score 过 `--min-score`（默认 70）的 item 汇总到 creator；creator 分 = 其 items 最高分；并列时按 `influence_score` 排序
4. **Pitch angles** — Sonnet 给 top N creator 各生成 2-3 个 angle

输出：terminal rich table + 可选 Markdown / **HTML 报告**（按 `--output` 后缀自动选）。

### `deep-dive <search_id> [--top 10] [--only id1,id2] [--skip id1] [--model ...]`

对已 search 的 top N creator 用 MiroThinker（或别的模型）做深度调研：
- **verified_active** — 确认 6 个月内还活跃
- **recent_themes** — 最近 6-12 月主流主题
- **sharp_quotes** — 2-3 句锋利引用 + 真实 URL + 日期
- **current_stance** — 当前立场总结
- **pitch_hook** — cold-email 开场白建议
- **contact** — email / Twitter / LinkedIn / contact form / preferred channel / 备注

**默认 MiroThinker**（`mirothinker-1-7-deepresearch`，~$0.30/creator，含 web search fee）。MiroThinker 跑一个 ~6-9 分钟。

**省钱档**：`--model anthropic/claude-sonnet-4.6` — best-effort from training data，不做 live search，~$0.02/creator，10 秒一个。Honest 模式：找不到不编造，sharp_quotes 留空。

`--only` 指定具体 creator_id 不受 `--top` 限制；`--skip` 排除已 deep-dived 的。

### `show <search_id> [--min-score 55] [--output file.html]`
重新渲染之前的 search，自动 join `pitch_angles` + `deep_dives`。`--output` 用 `.html` 后缀生成完整 HTML 报告。

### `status <creator_id> <campaign> <new_status> [--notes "..."]`
跟踪 outreach 状态。允许值：`not_contacted`、`pitched`、`replied`、`confirmed`、`declined`、`published`。`pitched` 和 `replied` 自动记时间戳。

### `discover-podcasts "<keyword>" [--limit 10]`
查 Apple Podcasts iTunes API，输出可粘贴到 YAML 的 stanza。不自动入库。

### `discover-substack <substack_url>`
抓某 Substack 的 `/recommendations` 页找候选。Best-effort，DOM 变就坏。

---

## 五、HTML 报告布局

每个 creator 卡片自上而下：

1. **Why we picked them**（绿色块）— 最高分 item + LLM 给的 reason
2. **Their recent work that matches the launch** — top 3 items, 每条点开就是文章/episode 链接
3. **Verified context**（紫色 = MiroThinker live search；灰色 = Sonnet best-effort）— recent themes / sharp quotes (带真实 source URL) / current stance / pitch hook
4. **How to reach them**（黄色块）— email / Twitter / LinkedIn / contact form / preferred channel
5. **Pitch angles**（橙色块）— 从 launch 数据生成的 2-3 个 angle

报告 dark-mode aware，无外部 JS / CSS 依赖。

---

## 六、数据模型（SQLite）

| 表 | 用途 |
|---|---|
| `creators` | seed / discovered creator，`UNIQUE(platform, handle)` |
| `items` | 90 天内 article / episode / video，`url` UNIQUE |
| `searches` | 每次 `search` 一行，含 extracted_topics |
| `relevance_scores` | 每次 search 每个 item 的 score + reason |
| `pitch_angles` | 每次 search 每个 creator 的 angles JSON |
| `deep_dives` | MiroThinker / Sonnet 深度调研的 payload + 用了什么模型 |
| `outreach` | 手动状态跟踪 `(creator_id, campaign)` UNIQUE |

数据库在 `./pitchfinder.db`（gitignored）。所有命令必须在项目根目录跑。

---

## 七、`.env.example` 完整模板

```bash
# --- OpenRouter (relevance scoring + pitch angles + lint) ---
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
PITCHFINDER_RELEVANCE_MODEL=deepseek/deepseek-chat-v3.1
PITCHFINDER_PITCH_MODEL=deepseek/deepseek-chat-v3.1
OPENROUTER_APP_NAME=pitchfinder
OPENROUTER_HTTP_REFERER=https://github.com/shanda-launchpad/shanda-pitchfinder

# --- MiroMind / MiroThinker (deep-dive enrichment) ---
MIROMIND_API_KEY=
MIROMIND_BASE_URL=https://api.miromind.ai/v1
MIROMIND_DEEPRESEARCH_MODEL=mirothinker-1-7-deepresearch
```

---

## 八、Out of scope（不构建）

- 邮件自动发送 / 自动 pitch
- 多用户 / auth / SaaS / 网页托管
- X / Twitter / LinkedIn / TikTok 内容抓取
- YouTube 视频 transcript 提取
- 中国大陆媒体（机器之心 / 量子位 / 36Kr 等）

---

## 九、踩过的坑（已修，留作记忆）

- **MiroThinker 默认 SSE streaming** — OpenAI SDK 默认 non-stream，必须用 `stream=True` 才能拿到 chunk
- **DeepSeek V4 / GLM 5.1 在 short JSON 输出场景下 OpenRouter 路由不稳** — 40-86% 返回空字符串
- **httpx 默认不带 brotli 解码器** — Accept-Encoding 不能写 `br`
- **HTTP 200 ≠ 内容真实** — `nopriors.com` 是 GoDaddy 待售页，所以加了 `pitchfinder lint`
- **podcast feed 的 `<id>` 字段有时是 UUID 不是 URL** — fetcher 已经只接 http(s) URL
