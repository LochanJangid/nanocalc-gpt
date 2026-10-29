# nanoCalc-GPT 🧮

**I taught a tiny GPT to do math. It never got told how addition works. Here's what went wrong, what fixed it, and the one trick that backfired.**

Best result: **99.51% exact match** on held-out `+ - * /` problems, with a model small enough to train on a single GPU.

---

## How this started

After building nanoGPT with Karpathy's Zero to Hero series, I wanted to try the exercise he leaves at the end of the GPT video: train a transformer on something simple and see what it actually learns.

I picked the first thing every kid learns: addition.

Our teachers explained the pattern and the mechanism. The model gets neither. All it has is a loss that says "wrong, try again." So the question became: can it discover carrying, on its own, from examples alone?

## Round 1: the model that couldn't carry

First attempt, I wrote problems the normal way: `11+06=17`.

It barely learned anything. Then I realized why. When we add, we start from the **right** (ones digit, then tens). A GPT writes **left to right**. So to write the first digit of the answer, the model needs the carry from a digit it hasn't written yet. There's no stable pattern to learn.

**Fix:** reverse the answer. `11+06=17` becomes `11+06=+7100`. Now the model writes the ones digit first, then the tens digit, and every digit only depends on stuff it has already seen.

Addition went to **100%**.

## Round 2: where does an example even end?

`1+2=3` and `0+15=15` have different lengths. In one long stream of text, the model had to figure out where each example starts before it could do any math.

**Fix:** make everything the same shape. Zero-pad the numbers, so every example is exactly 12 characters. Now each position always means the same thing.

## Round 3: stop predicting random numbers

The model was also being scored on guessing the two numbers and the operator. Those are random. There's nothing to learn there, only noise.

**Fix:** mask the loss so only the answer tokens count. The only thing left to learn is the actual relationship between the question and the answer.

## Round 4: all four operators

Then I fed it `-`, `*`, and `/` too, same model, same trick.

- Subtraction: easy, 100%.
- Multiplication: stuck, and the test score was way below the train score (62.8% test in an earlier run). The model was memorizing.
- **Fix:** weight decay. Penalizing big weights pushed it toward the general rule instead of a lookup table. Multiplication jumped to the mid-90s.

Then I made the model bigger, and it got interesting. More depth only helped if I also added width:

| Width / blocks | `*` | `/` | Overall |
|---|---|---|---|
| 64 / 4 | 96.00% | 78.95% | 98.30% |
| 64 / 6 | 81.30% | 82.46% | 93.56% |
| 128 / 4 | 96.60% | 84.21% | 98.59% |
| 128 / 6 | 99.50% | 80.70% | 99.48% |

Six blocks at width 64 was worse than four. A deeper, narrower model was harder to train, not smarter.

## Round 5: the trick that backfired

Division was stuck around 80% no matter how big the model got. Bigger model, same score. That told me it wasn't a capacity problem.

So I printed every wrong answer. All 12 mistakes were the **same digit**: the ones digit of the quotient, which is the very first digit the model writes. And every divisor was even or a multiple of 5 (2, 4, 5, 6, 8, 18).

My guess: reversal helps addition because the ones digit of a sum only depends on the ones digits you're adding. A quotient is the opposite. You work it out from the big end, like long division. By reversing it, I was making the model write the *hardest* digit first.

**Fix:** keep the division answer in normal order, reverse everything else.

Division went from 80.70% to **85.96%**. A small win, but it matches the story.

## Final scoreboard

Model: decoder-only transformer, 128 dims, 6 blocks, 4 heads, AdamW + OneCycleLR, weight decay 0.1, batch size 128, 30k steps.

| Operator | Test examples | Exact match |
|---|---|---|
| `+` | 1000 | 100.00% |
| `-` | 1000 | 100.00% |
| `*` | 1000 | 99.30% |
| `/` | 57 | 85.96% |
| **Overall** | 3057 | **99.51%** |

"Exact match" means every single answer token has to be right.

## What the data looks like

```
11+06=+7100   (11 + 6 = 17, answer reversed)
20-83=-3600   (20 - 83 = -63, answer reversed)
84/06=+0014   (84 / 6 = 14, answer NOT reversed)
```

## The honest part

I don't want you to leave with the wrong idea:

- **The test is small.** Operands are 00 to 99, so each operator has around 10,000 problems total. My test set is a random 10% of that. So this shows the model handles unseen *pairs*. It doesn't prove it learned the full algorithm. I haven't tested whether it works on 3-digit numbers (my guess: it won't, yet).
- **Division is only 57 test examples.** One example is worth 1.75 points, so be careful comparing those numbers.
- **Most numbers are from a single run.** I haven't done multi-seed runs yet.
- This hasn't been peer reviewed. It's one person learning in public.

## Run it yourself

```bash
git clone https://github.com/LochanJangid/nanocalc-gpt
cd nanocalc-gpt
pip install -r requirements.txt

python generate_data.py   # builds data/train.txt and data/test.txt
python train.py           # trains the model
python evaluate.py        # per-operator accuracy and the mistakes it made
```

## What I want to try next

- Train on 2-digit, test on 2-digit. Does it generalize, or just memorize better?
- Give multiplication a scratchpad (write out partial products first).
- Try an ablation for every trick above, to prove each one actually matters.

## Built with help from

- Andrej Karpathy's *Neural Networks: Zero to Hero*

---

Made by **Lochan Jangid** · [LinkedIn](https://linkedin.com/in/lochan-jangid) · [Portfolio](https://lochan.vercel.app)

If you see something I got wrong or know a better way, open an issue. I read all of them.