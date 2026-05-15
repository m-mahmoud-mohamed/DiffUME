# Dataset report — `umer1_sft_processed`

- Path: `/mnt/ceph-hdd/cold/nii00224/UME_R1/umer1_sft_processed`
- Total rows: **1,463,360**
- Columns: `['dataset_name', 'qry', 'pos']`
- Inspected rows per side: 200

## `dataset_name` distribution (first 20k rows)

- `llavahound_caption_retrieval`: 3520
- `llavahound_video_retrieval`: 3200
- `llavahound_qa`: 2688
- `vidore/colpali_train_set`: 1408
- `openbmb/VisRAG-Ret-Train-In-domain-data`: 960
- `VisDial`: 896
- `MSCOCO_i2t`: 896
- `MSCOCO_t2i`: 704
- `VisualNews_t2i`: 576
- `VisualNews_i2t`: 576
- `MSCOCO`: 576
- `Visual7W`: 561
- `ImageNet_1K`: 540
- `N24News`: 489
- `DocVQA`: 479
- `InfographicsVQA`: 320
- `CIRR`: 320
- `ChartQA`: 316
- `SUN397`: 236
- `WebQA`: 192
- `A-OKVQA`: 166
- `NIGHTS`: 128
- `HatefulMemes`: 96
- `VOC2007`: 93
- `OK-VQA`: 64

## QRY side

- sub-dict keys: `{'image': 200, 'video': 200, 'conversations': 200}`
- has `<think>` block (assistant): **100.0%**
- has `<answer>` block (assistant): **0.0%**
- has `<gen_emb>` token (assistant): **100.0%**
- has `<disc_emb>` token (user): **100.0%**
- has image: 100.0%
- has video (placeholder): 100.0%

### sample user turn

```
<image><video>
Find me an everyday image that matches the given caption: A meme of a cat sitting on the counter with carrots saying " I went to farmerz markit.
<disc_emb>
Represent the above input text, images, videos, or any combination of the three as embeddings. First output the thinking process in <think> </think> tags and then summarize the entire input in a word or sentence. Finally, use the <gen_emb> tag to represent the entire input.
```

### sample assistant turn

```
<think>So, let's analyze the input. The user's query is asking for a meme image. Key elements: meme, cat, sitting on counter, carrots, caption "I went to farmerz markit." So the main topic is a cat meme with carrots, caption about farmer's market. Need to identify the key components: the subject (cat), action (sitting on counter), objects (carrots), context (farmer's market joke, since "farmerz markit" is a play on "farmer's market" →往返谐音或改动), and the format is a meme. 

Synthesize into a single word or concise sentence. The essence is a meme featuring a cat on a counter with carrots, having a caption about a farmer's market. So maybe "cat meme carrots farmer's market caption". Wait, more concise: "cat meme carrots farmerz markit". Wait, need to capture all key elements. Alternatively, "meme with cat, carrots, farmerz markit caption". Hmm. Let's make it concise. The main elements: meme type, cat, carrots, caption about farmer's market (spelled playfully). So a good summary might be "cat meme with carrots farmerz market caption" or shorter like "cat candy carrots market meme" no, wait carrots with "farmerz markit" which is a typo maybe. Wait, the caption is "I went to farmerz markit." So the meme has a cat on a counter with carrots, caption about farmer's market (possibly with humorous spelling). So the key elements are: meme, cat, counter, carrots, caption "I went to farmerz markit". So the essence is a meme image with a cat on a counter, carrots, and that specific caption. So a concise summary would be "cat meme carrots farmerz market caption" or even shorter "cat meme carrots farm market". Wait, let's think step by step.

1. Main topic: finding an everyday image (meme) with cat, counter, carrots, caption "I went to farmerz markit." (which is a play on "farmer's market" → misspelled for humor).
2. Key entities: cat, carrots, counter, farmerz markit (caption), meme format.
3. Relationships: cat is on counter, next to carrots, caption is part of meme.
4. Context: eve
```

## POS side

- sub-dict keys: `{'image': 200, 'video': 200, 'conversations': 200}`
- has `<think>` block (assistant): **100.0%**
- has `<answer>` block (assistant): **0.0%**
- has `<gen_emb>` token (assistant): **100.0%**
- has `<disc_emb>` token (user): **100.0%**
- has image: 100.0%
- has video (placeholder): 100.0%

### sample user turn

```
<image><video>
Represent the given image.
<disc_emb>
Represent the above input text, images, videos, or any combination of the three as embeddings. First output the thinking process in <think> </think> tags and then summarize the entire input in a word or sentence. Finally, use the <gen_emb> tag to represent the entire input.
```

### sample assistant turn

```
<think>So, let's analyze the image. First, there's a Siamese cat sitting next to a pile of carrots with greens, probably parsley. The cat has a somewhat annoyed or disinterested expression. There are text overlays: "OH HAI" at the top right, "I WENT TO FARMERZ MARKIT" at the bottom. The background shows a kitchen area with a toilet, cabinets, and some red boxes. The main elements are the cat, carrots, text humorously mixing "farmer's market" into a往返 statement? Wait, "market" is spelled with a Z and "farm" with er, maybe a meme about going to the market and the cat's reaction. So key elements: Siamese cat, carrots, text overlays with play on words, kitchen setting. The humor is the cat saying it went to the farmers market, maybe not thrilled, and "OH HAI" as a greeting. So the essence is a humorous meme with a cat, carrots, and playful text.</think><answer>A Siamese cat with carrots, humorous text overlays about a farmers market in a kitchen setting.
<|begin_of_box|>A humorous meme featuring a Siamese cat with carrots and wordplay text<|end_of_box|>
<gen_emb>
```

## Phase 0b gate

- Reasoning-span coverage gate (≥80% on both sides for `<think>` and `<answer>`): **FAIL**
