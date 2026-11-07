# Demo video script (5 minutes): criterion-level clinical-trial screening system

Target length: about 5 minutes; the limit is 5–7. The optional segment brings it to about 6.
Columns: **time · what you do on screen · what you say.** Read the "say" text naturally; don't recite it.

---

## Before you press record (10 minutes of preparation)

1. **Plug in the charger.** Go to **Settings → System → Power → Power mode → Best performance**.
2. Close Chrome/Edge tabs you don't need, Kaggle, WhatsApp and anything else heavy.
3. Open **Notepad** and paste the demo patient below, so you can copy it during the recording.
4. Double-click `C:\dev\genai-a2\run_app.bat`. When the browser opens, run **one warm-up screening**:
   load any validation example and click *Screen patient*. The first run is always slower.
5. Set the browser zoom to **90%** (Ctrl and −) so the result table fits on screen.
6. Recorder: **Snipping Tool → Record** (Win+Shift+R), or **OBS Studio**. Select the whole screen and
   **turn the microphone on**. Do a 10-second test recording and listen back to it.
7. Leave the black `run_app.bat` window open behind the browser. You'll show it for 2 seconds.

### Demo patient (unseen; it is not in the dataset)

**Patient note**
```
A 62-year-old man with type 2 diabetes for ten years presents with burning pain in both feet. He takes metformin 1000 mg twice daily. His HbA1c last month was 8.4%. He had a myocardial infarction two years ago. He has never smoked.
```
**Inclusion criteria**
```
Adults aged 18 to 75 years
Diagnosis of type 2 diabetes mellitus
HbA1c between 7.0% and 10.0%
```
**Exclusion criteria**
```
Myocardial infarction within the past 6 months
Current smoker
Estimated GFR below 30 mL/min/1.73m2
Pregnant or breastfeeding women
```

What the fine-tuned model returns (checked; it is deterministic, so you'll see the same labels):

| Criterion | Prediction | Correct? |
|---|---|---|
| Age 18–75 | included [0] | ✔ |
| Type 2 diabetes | included [0] | ✔ |
| HbA1c 7–10% | not enough information [2] | ✘ 8.4% is in range; it even cites the right sentence |
| MI within 6 months | not excluded [3] | ✔ the MI was two years ago |
| Current smoker | not excluded [] | ✔ label, but cites nothing (sentence 4 says "never smoked") |
| eGFR < 30 | not excluded [] | ✘ kidney function is never mentioned, so it should be "not enough information" |
| Pregnant / breastfeeding | not applicable | ✔ the patient is a man |

Outcome: **Needs review (insufficient information)**, about 35 s on the laptop.

---

## The script

| Time | On screen | Say |
|---|---|---|
| **0:00–0:20** | Browser on the *Screen a patient* tab, nothing filled in. | "This is my Generative AI assignment: a system that screens a patient against a clinical trial's eligibility criteria one criterion at a time, and cites the sentences of the patient note that support each decision. It runs Qwen 2.5 1.5B, fine-tuned with LoRA, entirely on this laptop's CPU. It's an educational prototype, not a clinical decision tool." |
| **0:20–0:35** | Alt-Tab to the black `run_app.bat` window for 2 seconds, then back. | "It starts with a double-click on run_app.bat. No notebook or IDE needed, and it works offline once the models are downloaded." |
| **0:35–0:55** | Paste the **note**, the **inclusion** and the **exclusion** criteria from Notepad. | "This patient note and these criteria are new; they're not in the dataset. The note is plain free text: the app splits it into numbered sentences, because the model answers with sentence IDs as evidence." |
| **0:55–1:35** | Click **Screen patient**. Talk while the progress bar runs (about 35 s). | "Every criterion becomes a prompt with the note, the criterion type and the label set. The model must answer in JSON with a label and the supporting sentence IDs. The labels keep the dataset's categories, including *not enough information* and *not applicable*, so they're never forced into yes or no. To make this fast on a CPU, I wrote my own Transformer forward pass with a key-value cache. The note is processed once, and then all seven criteria are decoded together in one batch, each row of the cache with its own length. On this laptop that's four times faster than one criterion at a time." |
| **1:35–2:25** | Results appear. Point at the outcome box, then scroll the table row by row. | "The outcome is *needs review*. That comes from an explicit rule, not from the model, because the dataset only labels individual criteria. Looking at the rows: age and diabetes are *included*, citing sentence 0. The heart attack was two years ago, so the exclusion *within six months* is correctly *not excluded*, citing sentence 3. Pregnancy is *not applicable* for a man. There are two mistakes, and they're the same ones my error analysis found on the test set. HbA1c 8.4 is inside 7 to 10, but the model says *not enough information* while citing the right sentence: it struggles with numeric ranges. And kidney function is never mentioned, but it says *not excluded*: it reads missing information as a negative, which was its most frequent error. This is why every decision shows its evidence." |
| **2:25–2:45** | Click the **.html** report link under *Download report*; it opens in a new tab. Scroll to *Provenance*. | "Every run can be exported as an HTML or JSON report: the outcome, each decision with its evidence, the numbered note, and provenance — the model and adapter versions, the decoding settings, the code commit and a fingerprint of the input — plus the notice that this isn't a clinical decision." |
| **2:45–3:00** | Back in the app: select all the text in the note box, delete it and click **Screen patient**. An error message pops up. Paste the note back. | "Invalid input is handled. An empty note gives a clear message instead of a crash, and the app keeps working." |
| **3:00–3:50** | **Inference lab** tab. Leave the defaults and click **Run comparison** (about 30–40 s, talk over it). Then point at the *Tokens identical* column and at *Total s*. | "The inference lab runs the same prompt through each decoding method live. KV cache is my cached decoder. Speculative uses a fine-tuned 0.5B draft model that proposes four tokens, which the 1.5B model checks in one pass. HF generate is the library reference. All three produce identical tokens. Speculative decoding is faster here because on a CPU the cost of a step is reading the weights, and the draft is three times smaller. On the Kaggle T4 GPU the gain was only seven percent, and that comparison is in the report." |
| **3:50–4:10** | Scroll down, click **Run attention check** (instant). | "My tiled attention uses online softmax, like FlashAttention. Against a float64 reference all three kernels agree to about one in a million. On the GPU the tiled version used 13 times less memory at 2048 tokens, but it's slower because it isn't a fused kernel." |
| **4:10–4:40** | **Evaluation results** tab. Point at the test rows. | "On the held-out test split, which is eight unseen patients, fine-tuning raised accuracy from 0.33 to 0.74, macro-F1 from 0.26 to 0.55 and evidence F1 from 0.12 to 0.51. The test set was evaluated only once, after everything was frozen. The confusion matrix shows the main remaining error: *not enough information* predicted as *not excluded*." |
| **4:40–5:00** | Back to tab 1. | "To summarise: a small open model, fine-tuned in 16 minutes on a free GPU and running on a laptop, with its own KV cache, tiled attention and speculative decoding, all verified token by token. Again, this is an educational screening prototype, not a medical decision tool. Thank you." |

### Optional segment (+1 minute, total about 6): before vs. after fine-tuning, live
Insert after 2:45, before the inference lab.

| Time | On screen | Say |
|---|---|---|
| +0:00–0:50 | Open **Model and inference settings** → choose **Base model (before fine-tuning)** → **Screen patient** (about 45 s; talk over it). | "This is the same model before fine-tuning, with the LoRA update switched off. The weights stay in memory, so there's no second copy." *(When it finishes:)* "It wrongly excludes the patient for the old heart attack and labels five of seven criteria *not applicable*. Ignoring those would make the outcome look clean, so the app warns that most criteria were set aside and the result shouldn't be trusted. That safeguard came from watching this exact behaviour." |
| +0:50–1:00 | Switch the model back to **Fine-tuned (LoRA)**. | — |

---

## If something goes differently
- **A run is slow:** keep talking; the explanations above are written to fill the wait. Don't restart.
- **You stumble over a sentence:** pause for two seconds and say it again. You can trim it in Clipchamp, which
  comes with Windows 11 (open the video → drag the clip edges to cut).
- **Timings differ from the script:** that's fine. Say "about" and read the numbers on screen.

## After recording
1. Watch it once and check that the audio is clear and the text is readable.
2. Upload it to **YouTube as Unlisted** (or to Google Drive with *Anyone with the link can view*).
3. Open the link in a private window to check that it plays without signing in.
4. Send me the link and I'll add it to the README.
