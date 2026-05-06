# Project Narrative: Fairness in Text-to-Music Generation

---

## Layer 1 — The Hook: A Concrete Observation

While using Suno, a popular text-to-music generation model, we noticed something odd: prompting the model with *"make a waltz"* produced correct results far more reliably than *"make a song in 3/4 time"* — even though both descriptions refer to the exact same rhythmic structure. This wasn't an isolated glitch. The Reddit community around Suno had independently noticed the same pattern, and attempts to generate music in complex time signatures like 5/4 consistently collapsed into 4/4, the dominant meter of Western pop music.

This small observation opens a larger question: **who does this model serve well, and whose musical world does it erase?**

---

## Layer 2 — Two Fairness Problems

The observation above points to two distinct but related fairness issues:

**1. User-Experience Bias**
The model responds better to colloquial, culturally familiar descriptions ("waltz," "march," "jig") than to formal music theory terms ("3/4 time," "2/4 time," "6/8 time"). This creates a paradox: users with formal musical training — who are more likely to think and communicate in precise notational terms — are systematically disadvantaged compared to casual users who happen to know a genre name. A classically trained musician asking for "a piece in 3/4 time" may get worse results than someone who simply says "waltz."

**2. Cultural Bias**
When the model fails to generate non-4/4 time signatures and defaults to 4/4, it is not just making a technical error — it is erasing musical traditions. Countless musical cultures rely on non-4/4 meters: Bulgarian and Macedonian folk music in 7/8, Turkish aksak rhythms in 9/8, Dave Brubeck's jazz in 5/4, Irish jigs in 6/8. If the model consistently collapses these to 4/4, it implicitly treats Western pop music as the universal default and renders everything else invisible.

---

## Layer 3 — The Experiment

To test both biases systematically, we designed a controlled experiment across five time signatures (2/4, 3/4, 4/4, 5/4, 6/8) and two prompt styles:

- **Cultural prompts** — colloquial or genre-based descriptions that imply a time signature (e.g., "a lively Irish jig with fiddle and bodhrán")
- **Formal prompts** — structurally identical prompts that explicitly state the time signature (e.g., "a lively dance in 6/8 time with fiddle and bodhrán")

Each pair is matched for instrumentation, mood, and style — the only variable is how the time signature is described. We generate multiple outputs per prompt using two text-to-music models and measure the success rate: how often does the generated audio actually fall in the requested time signature?

The results can be read along two axes:

- **Vertically** (cultural vs. formal within the same time signature): reveals user-experience bias
- **Horizontally** (across time signatures within the same prompt style): reveals cultural bias

---

## Layer 4 — A Deeper Finding

Before running a single generation, the experiment design itself surfaces a striking observation: **we struggled to find widely recognized cultural names for 5/4.**

For 3/4, "waltz" is universally understood. For 6/8, "jig" and "tarantella" carry clear cultural meaning. For 2/4, "march" and "polka" are unambiguous. But for 5/4, the best references we could find were niche jazz knowledge ("Take Five") or a film theme ("Mission Impossible") — references far less accessible to a general audience.

This is not a flaw in our prompt design. It is itself a finding:

> *Unlike 3/4 (waltz) or 6/8 (jig), 5/4 lacks widely recognized cultural naming conventions — a reflection of its near-absence from Western popular music.*

The bias does not stop at the model's output layer. It has already permeated the language we use to talk about music. Users from traditions that rely on 5/4 or other asymmetric meters cannot leverage shared cultural vocabulary to describe what they want, because that vocabulary was never developed in mainstream Western culture to begin with.

---

## Layer 5 — The Structural Argument

This leads to the central claim of this project: **the fairness problem is not a technical accident — it is structural and self-reinforcing.**

```
Western pop music dominates training data
            ↓
4/4 is over-represented in the model
            ↓
Non-4/4 meters are rare in popular culture
            ↓
No widely shared cultural vocabulary develops for them
            ↓
Users cannot invoke them through cultural prompts
            ↓
The model receives even weaker training signal for these meters
            ↓
The bias deepens  ↑  (loop)
```

This feedback loop means the problem compounds over time. Every generation that collapses 5/4 to 4/4 reinforces the association. Every user who gives up on formal prompts and defaults to genre names feeds the cultural vocabulary gap.

---

## Broader Implications

The harms here are real and specific:

- **Musically trained users** are disadvantaged relative to casual users when precise notation is less effective than genre names.
- **Non-Western musical communities** — Balkan, Turkish, Indian classical, and others whose traditions depend on non-4/4 meters — find their musical heritage systematically excluded from AI-generated music.
- **Asymmetric meters broadly** are not just technically underserved; they are culturally unmarked in the Western vocabulary the model was trained on.

A text-to-music generation model that only reliably produces 4/4 music is not a neutral tool. It is one that encodes a particular cultural hierarchy and presents it as a universal default.

---

## Framing for This Course

This project sits at the intersection of two concepts central to responsible AI:

- **Representational harm**: the model fails to represent entire musical traditions, rendering them invisible in the generated output.
- **Allocative harm**: the model delivers unequal service quality to users based on their cultural background and musical training.

The deeper point is that these harms do not originate in the algorithm alone — they originate in the cultural power structures reflected in training data, and the algorithm amplifies and perpetuates them.
