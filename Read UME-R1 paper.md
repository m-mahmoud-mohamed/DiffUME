# Read UME-R1 paper

> let's first read this UME-R1 paper 18510_UME_R1_Exploring_Reasoni.pdf

<details><summary>8 previous messages</summary>

> I’ll first locate the PDF and extract enough text/metadata to understand its structure, then I’ll summarize the paper in a way we can use for follow-on code or research work. Commands I’m going to use: `rg --files` to find it, then a PDF text extractor if one is available.
>
> <details><summary>Explored 2 lists, ran 1 command</summary>
>
> - Ran `which pdftotext`
>
> <details><summary>Explored 2 lists</summary>
>
> - Listed files
> - Listed files
>
> </details>
>
> </details>
>
> The file is present at the repo root. `pdftotext` is not installed, so I’m checking for a Python PDF library next and will extract the paper text into the terminal only, without creating files.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `python3 -c "import os; p='18510_UME_R1_Exploring_Reasoni.pdf'; print(os.path.abspath(p)); print(os.path.getsize(p))"`
> - Ran `python3 -c "import importlib.util as u; print('pypdf', bool(u.find_spec('pypdf'))); print('PyPDF2', bool(u.find_spec('PyPDF2'))); print('fitz', bool(u.find_spec('fitz'))); print('pdfplumber', bool(u.find_spec('pdfplumber')))"`
>
> </details>
>
> There’s a bundled document/PDF runtime available in Codex, so I’ll use that instead of installing anything. I’m going to pull the page count and text outline first, then I’ll read the key sections in order.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `/Users/tianyuyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c "import importlib.util as u; print('pypdf', bool(u.find_spec('pypdf'))); print('PyPDF2', bool(u.find_spec('PyPDF2'))); print('fitz', bool(u.find_spec('fitz'))); print('pdfplumber', bool(u.find_spec('pdfplumber'))); print('pymupdf', bool(u.find_spec('pymupdf')))"`
> - Ran `/Users/tianyuyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c "import sys; print(sys.executable)"`
> - Ran `/Users/tianyuyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c "from pypdf import PdfReader; r=PdfReader('18510_UME_R1_Exploring_Reasoni.pdf'); print('pages', len(r.pages)); print('metadata', r.metadata); 
>   for i,p in enumerate(r.pages[:3],1):
>     t=p.extract_text() or ''
>     print('\n--- PAGE',i,'---')
>     print(t[:3500])"`
>
> </details>
>
> I’ve got the first pass: 28 pages, ICLR 2026, and the core mechanism is “generate reasoning + summary, then embed.” I’m extracting headings, tables, and experiment sections now so the summary doesn’t overfit to the abstract/introduction.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `/Users/tianyuyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c "from pypdf import PdfReader; import re; r=PdfReader('18510_UME_R1_Exploring_Reasoni.pdf');
> for i,p in enumerate(r.pages,1):
>     t=(p.extract_text() or '').replace('\n',' ')
>     heads=re.findall(r'(?:^| )((?:[1-9]|[A-Z])(?:\.[0-9A-Z]+)*\s+[A-Z][A-Z0-9 ,:;()\-–]+)', t)
>     print(f'--- PAGE {i} ---')
>     for h in heads[:8]: print(h[:180])
>     for key in ['Table ', 'Figure ', 'MMEB', 'reward', 'pass@', 'RL', 'GRPO', 'Results', 'Appendix']:
>         if key in t: print('contains', key)
> "`
> - Ran `/Users/tianyuyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c "from pypdf import PdfReader; r=PdfReader('18510_UME_R1_Exploring_Reasoni.pdf');
>   for i in range(3,10):
>     t=r.pages[i].extract_text() or ''
>     print('\n--- PAGE', i+1, '---')
>     print(t[:4500])"`
> - Ran `/Users/tianyuyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c "from pypdf import PdfReader; r=PdfReader('18510_UME_R1_Exploring_Reasoni.pdf');
>   for i in [15,16,20,21,24,27]:
>     t=r.pages[i].extract_text() or ''
>     print('\n--- PAGE', i+1, '---')
>     print(t[:4200])"`
>
> </details>
> </details>

I read the paper: [18510_UME_R1_Exploring_Reasoni.pdf](./18510_UME_R1_Exploring_Reasoni.pdf).

**High-Level Takeaway**

UME-R1 proposes a new kind of multimodal embedding model: instead of only encoding an input directly into an embedding, it can first generate reasoning and a concise summary, then use the generated content to produce a “reasoning-driven generative embedding.”

So the key shift is:

`input -> embedding`

becomes:

`input -> reasoning -> summary -> embedding`

The model still supports normal discriminative embeddings too.

**Core Method**

UME-R1 has two training stages:

1. **Cold-start SFT**
   - Starts from MMEB-V2 training data.
   - Uses a thinking-capable MLLM to annotate query-target pairs with reasoning and summaries.
   - Trains with three losses:
     - discriminative contrastive loss,
     - generative contrastive loss,
     - next-token prediction loss for reasoning/summary generation.

2. **RL with Verifiable Reward**
   - Uses GRPO.
   - Reward has two parts:
     - format reward: output must follow `<think>...</think><answer>...<gen_emb>`,
     - embedding reward: combines positive-vs-negative ranking and similarity gap.

**Results**

On MMEB-V2, across 78 image/video/visual-document tasks:

- UME-R1-2B scores **60.1 overall**.
- UME-R1-7B scores **64.5 overall**.
- It beats VLM2Vec-V2 while using less SFT data.
- Compared with DUME, the discriminative-only baseline trained under the same setup, UME-R1 gains strongly, especially on video and visual-document tasks.

A striking result: the oracle that chooses between discriminative and generative embeddings per example gets even higher scores, suggesting the two embedding modes are complementary.

**Important Caveat**

The generative embedding path is much slower.

From the appendix, on one L40s GPU:

- Generative embeddings: roughly **0.5 to 1.48 samples/s** depending on dataset.
- Discriminative embeddings: roughly **1.59 to 28 samples/s**.

So UME-R1 trades inference cost for better performance and interpretability.

**Why This Matters For Us**

This paper is very relevant to the repo name/theme: “Think to Embed.” The main implementation idea to look for next is probably whether this repo contains:

- the SFT data/template construction,
- embedding-token handling like `<disc_emb>` and `<gen_emb>`,
- contrastive loss over generated hidden states,
- GRPO/RL reward code,
- inference support for discriminative vs generative embeddings.

No files changed yet.

> currently this paper is based on autoregressive models to generate the reasoning trace. What about transfer this paradigm into the diffusion language model? How do you think this idea?

I think it’s a genuinely interesting idea, and not just “replace AR decoder with diffusion decoder.” The paradigm may actually become more distinctive in diffusion language models because diffusion gives you different inference knobs.

The UME-R1 recipe is:

`input -> reasoning trace -> summary -> embedding token hidden state`

For a diffusion language model, I’d reinterpret it as:

`input -> iteratively denoised latent/text rationale -> compressed semantic summary -> embedding state`

The promising parts:

1. **Reasoning as iterative refinement**
   Autoregressive CoT is left-to-right and often locks in early mistakes. A diffusion LM could revise the entire reasoning trace globally. That may be useful for embedding, where the final representation benefits from a coherent global semantic interpretation rather than a token-by-token story.

2. **Better inference-time scaling**
   UME-R1 shows pass@k improves with repeated sampling. Diffusion gives another scaling axis: more denoising steps, different noise schedules, self-consistency over multiple denoised traces, or ensemble embeddings across intermediate steps. This is a very natural fit.

3. **Embeddings from denoising trajectories**
   In AR UME-R1, the embedding comes from the hidden state at `<gen_emb>`. In a diffusion LM, you could use:
   - the final denoised sequence representation,
   - pooled hidden states from the summary span,
   - a learned `[EMB]`/`<gen_emb>` token after denoising,
   - or even aggregate states across denoising steps.

4. **Non-causal access may help retrieval**
   Diffusion language models can condition bidirectionally over the generated rationale. For retrieval/classification-style embeddings, bidirectional semantic consistency is often more important than causal generation fluency.

The hard parts are real, though:

1. **How to define the embedding state**
   AR models have a clean “last token hidden state” convention. Diffusion LMs may not have a natural final-token representation. You’d need a deliberate design: special embedding token, pooled rationale-summary states, or a projection over final denoised latents.

2. **Training objective mismatch**
   UME-R1 combines contrastive loss with next-token prediction. For diffusion LMs, the generative loss becomes denoising / masked reconstruction / score matching. The SFT objective might become:

   `L = L_disc_contrastive + L_gen_contrastive + L_denoise_reasoning`

3. **RL is less straightforward**
   GRPO over complete sampled rationales can still work, but token-level logprobs and KL are different for diffusion generation. You’d likely need diffusion-compatible policy optimization or a reward-weighted denoising objective.

4. **Latency may be worse**
   UME-R1 is already slower than discriminative embedding. Diffusion reasoning could be even more expensive unless you use few-step diffusion, latent diffusion, or generate very compact rationales.

My instinct: the strongest version is not “generate a long CoT with diffusion.” It is **diffusion-generated semantic refinement for embeddings**.

A good research direction could be:

**D-UME-R1: Diffusion Reasoning-Driven Multimodal Embeddings**

Core idea:

- Keep a discriminative embedding path.
- Add a diffusion reasoning path that denoises a compact rationale/summary representation.
- Use contrastive learning on the final denoised embedding token/state.
- Use reward or preference optimization to encourage denoising trajectories that improve positive-vs-negative retrieval margins.

The most elegant experiment would be small:

1. Start from UME-R1 data format.
2. Replace AR rationale generation with a masked/diffusion language modeling objective.
3. Generate only concise summaries first, not long CoT.
4. Compare:
   - discriminative embedding,
   - AR reasoning-driven embedding,
   - diffusion summary-driven embedding,
   - diffusion rationale+summary embedding.
5. Evaluate whether extra denoising steps improve pass@k or retrieval score.

My honest take: **yes, the idea is worth exploring**, but I’d avoid trying to mimic AR CoT too literally. Diffusion’s advantage is global revision and controllable iterative refinement, so the embedding paradigm should lean into that rather than forcing left-to-right reasoning into a different generator.

> in the template used by UME-R1, there is a gen_emb at the end of the respons and its final hidden state is used as the embedding. In the diffusion case, we could add this special token at anywhere right? Maybe add it in the prompt and after the denoising process is complete, we use this special token's hidden state as the embedding.

Yes, exactly. In a diffusion LM, the `<gen_emb>` token does not have to be “at the end” in the same way it does for an autoregressive model.

For AR UME-R1, `<gen_emb>` is placed at the end because the final hidden state can causally attend to:

`input + reasoning + summary`

So the last token is a convenient aggregation point.

For a diffusion language model, if the transformer is bidirectional or denoising-style, a special embedding token can be placed in the prompt, response scaffold, or another fixed slot, because after denoising its hidden state can in principle attend to the whole sequence.

A clean template could be:

```text
USER:
<image> <text/query> <disc_emb>
Represent the input as an embedding.

ASSISTANT:
<gen_emb> <think> [MASK/noise span] </think>
<answer> [MASK/noise span] </answer>
```

Then after denoising, use the final-layer hidden state at `<gen_emb>` as the generative embedding.

But I’d be careful about placement. The best location may depend on the attention mask and training objective.

Good candidate placements:

1. **Prefix embedding token**

```text
<gen_emb> <think> ... </think> <answer> ... </answer>
```

This makes `<gen_emb>` a global readout token, like `[CLS]`. Very natural for bidirectional diffusion.

2. **Between reasoning and summary**

```text
<think> ... </think> <gen_emb> <answer> ... </answer>
```

This may encourage it to integrate reasoning before summary, but it is less clean semantically.

3. **Suffix embedding token**

```text
<think> ... </think> <answer> ... </answer> <gen_emb>
```

This mirrors UME-R1 and is safest if the model still has any causal or semi-causal bias.

4. **Prompt-side embedding token**

```text
USER:
<image> <query> <gen_emb>
ASSISTANT:
<think> ... </think> <answer> ...
```

This is plausible only if the prompt token’s representation is updated through full bidirectional denoising attention. If prompt tokens are frozen or only used as conditioning keys/values, this may not work.

My preferred first experiment would be:

```text
USER:
<image> {query/target}
Represent the above input as an embedding.

ASSISTANT:
<gen_emb> <think> {reasoning} </think>
<answer> {summary} </answer>
```

Use `<gen_emb>` as a readout token whose hidden state is trained contrastively.

Training objective:

```text
L = L_denoise(reasoning, summary)
  + L_gen_contrastive(h_<gen_emb>)
  + optional L_disc_contrastive(h_<disc_emb>)
```

One nice trick: keep `<gen_emb>` unnoised and only noise the reasoning/summary spans. That way it acts as a stable aggregation token while the model learns to denoise the explanatory content around it.

The key question is whether `<gen_emb>` should be **denoised as a generated token** or **kept fixed as a readout token**. I’d start with fixed/unmasked `<gen_emb>`, because then the contrastive loss always lands on a stable position. Later you can test noising it too.

> i tthink we should not mask it and just keep it as the other part of the prompt.

I agree. Keeping `<gen_emb>` unmasked as part of the prompt is probably the cleaner design.

Then it behaves less like a token the model has to “generate” and more like a **readout anchor**. The diffusion process refines the reasoning/summary tokens around it, while `<gen_emb>` stays stable and collects information through attention.

A possible template:

```text
USER:
<image> {query_or_target}
Represent the above input as an embedding.

ASSISTANT:
<gen_emb> <think> [noised reasoning span] </think>
<answer> [noised summary span] </answer>
```

During training:

- Keep prompt tokens unmasked/unnoised:
  - image/text input,
  - instruction,
  - `<gen_emb>`,
  - structural tags like `<think>`, `</think>`, `<answer>`, `</answer>` maybe.
- Diffuse only the semantic generation spans:
  - reasoning tokens,
  - summary tokens.

Then compute:

```text
h_gen = final_hidden_state(position_of_<gen_emb>)
```

and apply contrastive learning:

```text
L_gen_ctr = InfoNCE(h_gen(query), h_gen(target))
```

This has a nice interpretation:

`<gen_emb>` is a fixed semantic collector conditioned on the denoised reasoning state.

One subtle point: if `<gen_emb>` appears before the reasoning/summary, the model must let that token attend bidirectionally to later tokens. So this works best with a diffusion/bidirectional attention setup. If the architecture still uses causal attention in the decoder, suffix placement is safer.

My preferred initial version:

```text
USER:
<image> {query_or_target}
Represent the above input text, image, video, or document as an embedding.

ASSISTANT:
<gen_emb> <think> {reasoning} </think>
<answer> {summary} </answer>
```

Loss:

```text
L = L_diffusion(reasoning, summary)
  + λ_g L_gen_ctr(h_gen)
  + λ_d L_disc_ctr(h_disc)
```

And the ablation I’d absolutely run:

- `<gen_emb>` before `<think>`
- `<gen_emb>` after `</answer>`
- no reasoning, summary only
- reasoning + summary
- masked/noised `<gen_emb>` vs fixed `<gen_emb>`

My bet: fixed prefix `<gen_emb>` will be strong if the diffusion LM is truly bidirectional. It gives you a CLS-like embedding token whose representation improves as the denoising context becomes semantically sharper.

> you can see there is a SDAR-VL directory, it's the SoTA multimodal understanding diffusion language model. you can go though it first. Maybe we can build this diffusion generative embedding model by adapt this codebase

<details><summary>6 previous messages</summary>

> Great, I’ll inspect `SDAR-VL` as the candidate backbone/codebase and map where generation, diffusion/noising, model forward hidden states, and training objectives live. No edits yet; I’m just building the mental wiring diagram before we touch anything.
>
> <details><summary>Explored 1 search, 3 lists</summary>
>
> - Listed files in `SDAR-VL`
> - Listed files in `SDAR-VL`
> - Listed files in `SDAR-VL`
> - Searched for `diffusion|mask|denois|generate|forward|hidden_states|logits|loss|train|attention|Qwen|CLIP|vision|image` in `SDAR-VL`
>
> </details>
>
> The repo is a thin top-level wrapper plus a modified `llama_factory_sdar` training tree. The first important signal: SDAR-VL uses block-wise discrete diffusion with causal dependencies between blocks, so `<gen_emb>` placement will interact with the block attention mask rather than being completely “anywhere.”
>
> <details><summary>Explored 3 files, 3 searches, 1 list, ran 1 command</summary>
>
> <details><summary>Explored 3 files, 3 searches, 1 list</summary>
>
> - Read `./SDAR-VL/README.md`
> - Read `./SDAR-VL/generate.py`
> - Listed files in `llama_factory_sdar`
> - Read `./SDAR-VL/training/README.md`
> - Searched for `*.yaml` in `llama_factory_sdar`
> - Searched for `inference/sdar_v_8b/(vocab|merges)`
> - Searched for `mask_id|block_diffusion|effective|beta|rectify|sdar|store_kv|MASK|diffusion|denoising|num_transfer|labels|ignore_index|compute_loss|prediction_step|loss_mask` in `llama_factory_sdar`
>
> </details>
>
> - Ran `git -C SDAR-VL/training/llama_factory_sdar status --short`
>
> </details>
>
> I found the custom training knobs: `noise_scheduler_type`, beta scheduler parameters, and SDAR-specific templates/plugins. Next I’m reading the SFT trainer and SDAR model forward because that tells us exactly where a contrastive embedding loss could attach.
>
> <details><summary>Explored 21 files, 9 searches</summary>
>
> - Read `./SDAR-VL/training/llama_factory_sdar/src/llamafactory/hparams/finetuning_args.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/src/llamafactory/train/sft/trainer.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/src/llamafactory/train/sft/trainer.py`
> - Searched for `def forward|hidden_states|last_hidden_state|output_hidden_states|loss|mask_id|labels|attention_mask|store_kv` in `modeling_sdar.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_sdar.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_sdar.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/src/llamafactory/data/mm_plugin.py`
> - Searched for `sdar_v\s*=|Template\(|register_template|SDARVPlugin|<\|MASK\|>|mask` in `data`
> - Searched for `def get_batch_loss|noise_scheduler_type|mask_id|MASK|labels.*mask|input_ids.*mask|training_step|scheduler_kwargs|beta|CrossEntropy|torch.where` in `modeling_sdar.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/fused_linear_diffusion_cross_entropy.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/src/llamafactory/data/mm_plugin.py`
> - Searched for `fuse_cross_entropy|FusedLinear|Diffusion|diffusion_cross|noise_scheduler|training_step|scheduler_kwargs|mask_id|151669|labels` in `llamafactory`
> - Searched for `register_template\(.*sdar|name=\"sdar_v\"|sdar_v` in `template.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_llava_onevision.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/src/llamafactory/data/template.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/examples/train_full/sdar_vl_4b/sdar_v_full_sft_stage4_1.yaml`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/fused_linear_diffusion_cross_entropy.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_llava_onevision.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_llava_onevision.py`
> - Searched for `class .*Onevision|def forward|p_mask|logits_to_keep|prompt_mask|mask_token_id|training_step|scheduler_kwargs|noise` in `modeling_llava_onevision.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_llava_onevision.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_llava_onevision.py`
> - Searched for `calculate_token_nums|def block_attn_mask|modify_padded_position_ids_2d|noise_scheduler_type|scheduler_kwargs` in `modeling_llava_onevision.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_llava_onevision.py`
> - Searched for `scheduler_kwargs|noise_scheduler_type|training_step` in `modeling_llava_onevision.py`
> - Searched for `block_size|block_length` in `llama_factory_sdar`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_llava_onevision.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/src/llamafactory/hparams/data_args.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/src/llamafactory/data/processor/supervised.py`
> - Read `./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_llava_onevision.py`
>
> </details>
> </details>

I went through `SDAR-VL`. Yes, this codebase is a plausible base for a diffusion generative embedding model, but the design has one important constraint: SDAR-VL is **block-wise diffusion**, not fully bidirectional over an arbitrary whole sequence.

The key files are:

- [generate.py](./SDAR-VL/generate.py): block-wise diffusion inference.
- [modeling_llava_onevision.py](./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_llava_onevision.py): multimodal SDAR-VL model, noise injection, hidden states.
- [modeling_sdar.py](./SDAR-VL/training/llama_factory_sdar/inference/sdar_v_8b/modeling_sdar.py): SDAR language backbone.
- [trainer.py](./SDAR-VL/training/llama_factory_sdar/src/llamafactory/train/sft/trainer.py): SFT training loop.
- [supervised.py](./SDAR-VL/training/llama_factory_sdar/src/llamafactory/data/processor/supervised.py): labels/prompt masking.
- [template.py](./SDAR-VL/training/llama_factory_sdar/src/llamafactory/data/template.py): `sdar_v` chat template.

My read: your idea of keeping `<gen_emb>` unmasked as part of the prompt is exactly compatible with SDAR-VL’s training code, because SDAR-VL defines prompt positions as `labels == -100`, and `forward_add_noise_packed(...)` avoids noising those positions.

The adapted template should probably be:

```text
<|im_start|>user
<image> {query_or_target}
Represent the input as an embedding. Generate reasoning and a concise summary.
<|im_end|>
<|im_start|>assistant
<gen_emb> <think> {reasoning} </think>
<answer> {summary} </answer>
<|im_end|>
```

Here `<gen_emb>` should be in the assistant prefix but labeled `-100`, so it is **not noised** and **not predicted**. Reasoning and summary are normal target tokens, so SDAR-VL diffuses/denoises them.

One subtlety: because SDAR-VL uses block-wise causal structure, I would initially put `<gen_emb>` **inside the first assistant block, before `<think>`**, but test suffix placement too. Prefix placement is elegant, but the attention mask may make later denoised content influence it differently during training/inference than in a fully bidirectional diffusion LM.

A practical adaptation path:

1. Add special tokens: `<gen_emb>` and optionally `<disc_emb>`.
2. Build UME-style paired data: query and target each contain reasoning + summary.
3. Modify the data processor so `<gen_emb>` is prompt/readout token with label `-100`.
4. In model forward, request/use final hidden states and gather hidden state at `<gen_emb>`.
5. Add InfoNCE contrastive loss on normalized `<gen_emb>` hidden states.
6. Keep SDAR-VL’s diffusion CE loss on reasoning/summary tokens.
7. Later add discriminative `<disc_emb>` and maybe RL/reward training.

So the SFT objective becomes:

```text
L = L_diffusion_reasoning_summary
  + lambda_gen * L_contrastive(h_<gen_emb>)
  + optional lambda_disc * L_contrastive(h_<disc_emb>)
```

I think this is a strong research direction. The first prototype should avoid RL and just prove the central claim: **does a fixed, unnoised diffusion readout token become a better multimodal embedding when surrounded by denoised reasoning/summary tokens?**

> in this blockwise diffusion model, would a special token in the prompt attend to all the generated reasoning content?

Short answer: **no, not if `<gen_emb>` is placed in the prompt before the generated reasoning span**.

In SDAR-VL’s current block-wise diffusion setup, attention is still **block-causal across blocks**:

- tokens in a block can interact within that block,
- later blocks can attend to earlier blocks,
- earlier blocks do **not** attend to later generated blocks.

So a prompt-side `<gen_emb>` would help condition the generated reasoning, but its hidden state would not naturally become a readout over all later reasoning content.

This is different from a fully bidirectional diffusion LM.

For our embedding design, that means the safer choices are:

1. **Put `<gen_emb>` after reasoning/summary**

   ```text
   <think> ... </think>
   <answer> ... </answer>
   <gen_emb>
   ```

   Keep `<gen_emb>` unmasked / label `-100`, but place it at the end. Then its hidden state can attend to preceding reasoning and summary under the block-causal mask.

2. **Two-pass readout**

   First generate/denoise reasoning and summary. Then run a second forward pass over:

   ```text
   input + reasoning + summary + <gen_emb>
   ```

   and use the hidden state at `<gen_emb>`. This is closest to UME-R1 conceptually and avoids fighting the generation loop.

3. **Custom infilling-style diffusion**

   Build a sequence like:

   ```text
   prompt + [MASK reasoning/summary span] + <gen_emb>
   ```

   and modify generation so fixed unmasked tokens can appear after masked spans. The current `block_diffusion_generate` assumes generation happens after the prompt, so this would require more surgery.

My recommendation: for SDAR-VL, use **suffix `<gen_emb>`** or **two-pass readout**. If we want the cleanest first prototype, I’d choose two-pass readout because it preserves your idea that `<gen_emb>` is unmasked while still guaranteeing it sees the completed reasoning content.

> but this infilling-style diffusion is more attractive right? It will join the whole denoise process

Yes. Infilling-style diffusion is more attractive scientifically.

It makes `<gen_emb>` part of the same denoising field, instead of being a readout bolted on after generation. That is much closer to the idea of **diffusion reasoning-driven embedding**, where the representation emerges while the rationale is being refined.

The attractive version is:

```text
USER:
<image> {query_or_target}
Represent the input as an embedding.

ASSISTANT:
<gen_emb> <think> [MASK ... MASK] </think>
<answer> [MASK ... MASK] </answer>
```

or even:

```text
USER:
<image> {query_or_target}
Represent the input as an embedding.

ASSISTANT:
<think> [MASK ... MASK] </think>
<answer> [MASK ... MASK] </answer>
<gen_emb>
```

But the key is: `<gen_emb>` is **fixed/unmasked**, while reasoning and summary are denoised around it.

If the attention mask is adjusted so fixed tokens and noised tokens can attend inside the same denoising region, then yes, `<gen_emb>` participates throughout the denoising process. Its hidden state changes as the surrounding masked rationale becomes cleaner.

That gives us several nice properties:

- `<gen_emb>` can see partially denoised reasoning at every step.
- reasoning tokens can also condition on `<gen_emb>`, so the embedding objective can shape the reasoning trajectory.
- the contrastive loss can be applied during training on the same forward pass as denoising.
- at inference, more denoising steps may genuinely improve the embedding, not only the text.

The required change is mainly conceptual but nontrivial in SDAR-VL:

Current SDAR-VL generation is shaped like:

```text
[prompt tokens][generated block 1][generated block 2]...
```

For infilling, we want something like:

```text
[condition/prompt tokens][fixed readout token][masked reasoning/summary slots]
```

or:

```text
[condition/prompt tokens][masked reasoning/summary slots][fixed readout token]
```

That means we need a generation mode where:

- the full sequence layout is known before denoising,
- some positions are fixed/unmasked,
- some positions are mask tokens,
- the model iteratively updates only mask positions,
- the attention mask lets `<gen_emb>` and the masked rationale communicate appropriately.

For SDAR-VL specifically, I’d prototype the least invasive version first:

```text
[prompt][<gen_emb>][<think>][MASK...][</think>][<answer>][MASK...][</answer>]
```

Then modify generation so `x` is initialized with the full scaffold, not just prompt + all masks. Only the reasoning/summary slots are updateable.

The crucial experiment is attention:

- If we keep SDAR’s original block-causal mask, prefix `<gen_emb>` may still not attend to later masked slots.
- So for the infilling region, we likely need a **bidirectional/block-diagonal denoising mask** among `<gen_emb>`, reasoning, and summary slots.
- The prompt can remain causal/conditioning-only.

So the attention structure should be roughly:

```text
prompt -> prompt
denoising region -> prompt + denoising region
```

And inside the denoising region, use bidirectional attention.

This is the more interesting model. It is also more of a paper-worthy departure from UME-R1 than simply replacing AR generation with SDAR-VL.