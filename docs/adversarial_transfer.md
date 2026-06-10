# Why the adversarial image stops working outside the original attack

The proof-of-concept reliably flips the model's answer **inside the attack loop**, yet the
saved `adversarial.png` usually behaves like a normal image when you open it in
`chat_with_image.py` and ask a different question. That is expected, and it is worth
understanding why before trying to fix it.

## What the attack actually optimizes

`run_poc.py` does **not** make the model "think a cat is a dog" in any general sense. It:

1. Resizes the pixel tensor to a fixed `448x448` (`_prepare_images`, bilinear).
2. Runs the Qwen processor (smart-resize to the patch grid, normalize, patchify).
3. Reads the **vocabulary logits at the single final prompt position** — the first answer token.
4. Slices those logits down to the allowed letter tokens (`A`, `B`, ...).
5. Pushes the perturbation, by gradient sign steps inside an `epsilon=8/255` L∞ ball, so that
   the letter for the wrong/target class wins **at that one position, under that one prompt**.

So the optimization target is razor-thin: one token position, a handful of letter logits, one
fixed prompt, one fixed preprocessing pipeline. The perturbation is a high-frequency pattern
tuned to that exact computation graph — not a robust, semantically meaningful change to the
picture.

## Why it does not transfer

**1. Preprocessing mismatch (only in `--native-preprocessing` mode now).**
The attack feeds the model a tensor that was bilinearly resized to `448x448` with
`do_rescale=False`. If the image is instead opened with PIL at its original resolution and run
through the processor's *default* smart-resize and `/255` rescale, you get a different resize
target, resampling, and patch grid → the carefully aligned perturbation lands on different
pixels and is largely averaged away. Adversarial perturbations are notoriously fragile to
resize/resample/crop. The chat script now shares the attack's exact preprocessing **by
default**, so this factor is removed unless you explicitly pass `--native-preprocessing` — which
is also the realistic threat model (a victim opening the file normally).

**2. 8-bit quantization on save.**
`epsilon = 8/255` is only ~2 levels of an 8-bit channel. Saving to PNG rounds the float
perturbation to `uint8`, and any resampling afterward smears those ~2 levels across
neighboring pixels. A large share of the signal is gone before the model ever sees it.

**3. The objective is the first letter token, not the model's understanding.**
The loss only moves letter-token logits at one position. Under "Describe this image" there is
no letter to pick; the gradient direction that flipped `A`→`B` has little leverage on a
free-form description. The attack never optimized the model's visual *features*, only a narrow
output projection conditioned on the multiple-choice prompt.

**4. Prompt / chat-template conditioning.**
The first-token distribution depends on the entire preceding context (system text, chat
template, the exact multiple-choice wording). Gradients were taken at *that* context. Change
the prompt and you change the function being attacked, so the perturbation is off-target.

**5. No robustness built into the perturbation.**
Vanilla PGD/APGD finds the *thinnest* perturbation that works for the exact forward pass. With
no augmentation during the attack, there is no pressure for it to survive any of the
transformations above.

## How to confirm the diagnosis quickly

The chat script preprocesses images through the **same shared code path** as `run_poc.py` by
default (`image_to_tensor` + `prepare_image_batch`: fixed `448` resize, `do_rescale=False`), so
item 1 is removed and the prompt is the only difference:

```bash
# attack-matched preprocessing is the default; keep --resize the same as the attack run
python scripts/chat_with_image.py outputs/demo/adversarial.png \
  --prompt "Choose exactly one option and answer with only the letter.\nA: cat\nB: dog"
```

- Same prompt as the attack, matched preprocessing, **still not fooled** → the gap is PNG
  `uint8` quantization (item 2): the attack optimized the float tensor, the chat loads the
  rounded image. Confirm by lowering the bar to the saved file in the attack too.
- Matched preprocessing + original prompt fools it, but a **different prompt** does not → items
  3–4 (narrow first-token objective and prompt conditioning).
- Pass `--native-preprocessing` to add item 1 back (original resolution + processor rescale) and
  see how much the resize/resample path alone costs.

## How to get an image that transfers

Roughly in order of impact:

1. **Expectation over Transformations (EOT).** Average the gradient over random
   preprocessings each step: random resize sizes, small rotations/crops, JPEG/quantization,
   brightness/contrast jitter. The perturbation then has to work across a *distribution* of
   pipelines instead of one. The APGD code already exposes an `eot_iter` loop — apply a fresh
   random transform inside each EOT iteration rather than recomputing the same forward.

2. **Quantization-aware attack (straight-through estimator).** Simulate the `uint8` round trip
   in the forward pass so the perturbation survives saving:
   `x_q = x + (torch.round(x * 255) / 255 - x).detach()`. Optimize on `x_q`, keep the real
   gradient flowing to `x`. Also evaluate the attack on the *actually-saved* image (save → load
   → forward) inside the loop, not just on the live tensor.

3. **Prompt ensembling.** Sum the loss over several prompts (the multiple-choice prompt plus
   "What is in the image?", "Describe the image.", etc.) so the perturbation is not tied to one
   context. This directly targets cross-prompt transfer.

4. **Attack a broader, earlier target than the first letter.** Instead of (or in addition to)
   letter logits, push the model's **image/feature representation** toward the target class
   (e.g. maximize the likelihood of a full target sentence across prompts, or match the
   hidden-state/vision-embedding of a reference target image). Effects on internal features
   generalize across prompts far better than a single output-token margin.

5. **Match the eval pipeline, or remove the double resize.** The attack currently resizes to
   `448` *and* lets the processor resize again. Optimize in the space the model actually
   consumes (feed the processor's target size directly, or include the full processor in the
   loop — it already is differentiable here), and craft/evaluate the perturbation at the same
   resolution you will load it at.

6. **Spend more budget if the threat model allows it.** A larger `epsilon` and more steps make
   a more robust (and more visible) perturbation; transfer generally needs a stronger signal
   than a single white-box hit.

A practical recipe: EOT (resize + JPEG + quantization) **plus** quantization-aware steps
**plus** prompt ensembling, evaluated on the saved file, is usually enough to get a perturbation
that still bites under a different prompt and a normal image-loading path.
