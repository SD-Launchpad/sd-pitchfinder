# PitchFinder

给一个产品发布,自动找出**最值得 pitch 的 newsletter / podcast / YouTuber / 博客**,并产出带联系方式、近期作品和 pitch angle 的分层名单。

---

## 这是做什么的 / 解决什么问题

发布一个新产品时,你想找媒体/创作者帮你传播。手工找的痛点是:

- 不知道**去哪找**对题的独立创作者;
- 找到一堆,但分不清谁是**真·中立 thought leader**,谁是内容农场 / 竞品 / 自家带货的商业站;
- 找到了也不知道**怎么联系**、**该说什么**。

PitchFinder 把这件事自动化:**全网发现 → 打分 → 分层 → 深度核验 → 出名单**。一条命令,产出可以直接拿去建联的 A/B 分层报告(HTML + CSV + Markdown)。

## 谁用

你自己(或同事)——给任意一个产品/品牌跑一次,拿到一份 outreach 名单。每个品牌一个配置文件,可重复复用。

---

## 你要给的 input

只有一个:**一个品牌配置文件 `brands/<brand>.yaml`**。里面写清楚:

| 字段 | 说明 |
|---|---|
| `brand` | 品牌名(报告文件名用) |
| `one_liner` / `positioning` | 产品是做什么的、怎么定位(喂给打分和分层) |
| `themes` | 关键词主题(全网发现就按这些去搜,前几个最重要) |
| `competitors` | 竞品名(分层时会自动 drop 掉竞品及其自营媒体) |
| `do_not` | 红线(pitch angle 里不能编造的东西) |
| `platforms` | 要找的平台:substack / blog / podcast / youtube |

照着 `brands/apodex.yaml` 抄一份改即可。

> 一次性配置:把几个 API key 填进 `.env`(见最下)。

## 一条命令跑完

```bash
pitchfinder campaign brands/<brand>.yaml
```

约 30–45 分钟(大头是深度核验)。想省钱/提速:`--budget 5`(限制深验数量)、`--skip-discovery`(复用已有库)。

## 中间发生了什么(5 步)

```
1. 发现   discover-web —— 用 Brave + Querit 全网搜,按 themes 捞候选创作者,解析出他们的 RSS feed
2. 抓取   refresh —— 拉每个创作者自己 feed 的近 90 天真实发文
3. 打分   search —— 用便宜模型给每篇内容打 0-100 相关性分,聚合到创作者
4. 分层   classify —— 用强模型(Sonnet)判 A / B / drop:
            · drop 内容农场、竞品、自家带货的商业站(只留中立第三方)
            · A = 强烈推荐建联,B = 可以建联
5. 深验   deep-dive —— 用 Apodex 对 Tier-A 头部做真实 web 搜索,
            拿到 verified 联系方式 + 近期原话 + pitch hook
```

## 你拿到的 output

`reports/<brand>-<日期>.html` / `.csv` / `.md` 三份同内容:

- **A/B 分层名单**(A 在前,drop 的不出现);
- 每个创作者一张卡片:为什么选他 → 匹配的近期作品(带链接)→ 怎么联系 → 2-3 条贴你产品的 pitch angle;
- 头部若干个有 Apodex **深度核验**的联系方式和原话;
- **CSV** 是扁平表,直接拖进 Google Sheet 跟进 outreach。

---

## 单独命令(也可不跑整条 campaign)

| 命令 | 作用 |
|---|---|
| `discover-web "<themes>" [--brand b.yaml]` | 全网发现创作者,产候选 YAML(无 key 的源自动跳过) |
| `load <yaml>` / `refresh` | 入库 / 拉 feed |
| `search "<描述>" [--output x.csv]` | 打分排序,可直接导 CSV/HTML/MD |
| `classify <search_id> --brand b.yaml` | 自动分 A/B/drop |
| `tier <search_id> <creator_id> <A\|B\|drop>` | 人工调档(不会被自动分层覆盖) |
| `deep-dive <search_id> --only <ids>` | 对指定创作者做深度核验 |
| `show <search_id> --output x.html` | 重新渲染报告(html/csv/md 按后缀) |

## 配置 / Key(一次性,填进 `.env`)

```bash
cp .env.example .env   # 然后在 .env 里填:
OPENROUTER_API_KEY=     # 打分 + pitch angle + 分层(必填)
APODEX_API_KEY=         # 深度核验(深验需要)
BRAVE_API_KEY=          # 全网发现(主)
QUERIT_API_TOKEN=       # 全网发现(补);Brave/Querit 有一个就能用
```

> key 在 Terminal 里填,别贴进聊天。`.env` 已 gitignore。

## 成本

单轮 campaign ≈ **$1.5–3**(发现/打分很便宜,大头是 Tier-A top-N 的 Apodex 深验)。深验贵的留到最后、只跑头部,所以可控。

## 测试

```bash
uv pip install -e ".[dev]"
pytest
```

---

旧版命令细节(model lineup、各源行为、踩过的坑)见 git 历史。当前默认形态就是上面的 `campaign` 漏斗。
