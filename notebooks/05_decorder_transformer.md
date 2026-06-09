# Step 05 - Decoder Transformer Based NLG Model

## Controlled Natural Language Generation using Custom Transformer

---

# 1. Objective

The objective of this module is to build a custom Transformer based Decoder model for **Controlled Natural Language Generation (NLG)**.

The model learns:

```
Intent + Slots  --->  Decoder Transformer  ---> Natural Language Response
```

Example:

## Input

```
[BOS] 
intent : order_status 
<sep> 
order_id : ORD841324 
<sep> 
channel : website 
[EOS]
```

## Output

```
I can help with order status for order ORD841324.
```

The decoder should learn:

- Intent specific responses
- Sentence structure
- Slot usage
- Domain language

---

# 2. Position in Complete Assignment Pipeline

```
Dataset Generation
        |
        |
Schema Creation
        |
        |
Train / Validation / Test Data
        |
        |
Linearization
        |
        |
[BOS] intent + slots [EOS]
        |
        |
Vocabulary Creation
        |
        |
Text Tokens -> Token IDs
        |
        |
============================
 STEP 05 Decoder Transformer
============================
        |
        |
Embedding Layer
        |
        |
Positional Encoding
        |
        |
Transformer Blocks
        |
        |
Generated Response
```

---

# 3. Input Data

The model consumes the output generated from Step 03.

Example JSON:

```json
{
    "linearized_sequence":
    "[BOS] intent : refund_status <sep> order_id : ORD123 [EOS]",

    "target_text":
    "I can help with refund status for order ORD123."
}
```

The learning problem:

```
Input Sequence
        |
        |
        v
Target Response
```

---

# 4. Tokenization

Neural networks cannot process raw text.

Text must be converted into numbers.

Example:

Vocabulary:

```json
{
    "[PAD]" : 0,
    "[BOS]" : 1,
    "[EOS]" : 2,
    "order_status" : 25
}
```

Text:

```
order_status
```

becomes:

```
25
```

---

# 5. Special Tokens

## PAD - Padding Token

Purpose:

Make every sequence the same length.

Example:

Original:

```
I need refund
```

After padding:

```
I need refund PAD PAD PAD
```

Deep learning models process fixed size tensors.

---

## BOS - Beginning Of Sequence

Indicates where generation starts.

Example:

```
[BOS] Your order is shipped
```

---

## EOS - End Of Sequence

Indicates generation completion.

Example:

```
Your order is shipped [EOS]
```

---

# 6. Dataset Conversion

Before training:

Text:

```
intent order_status order ORD123
```

is converted into:

```
[15, 25, 40, 101]
```

Then converted into PyTorch tensor:

```python
torch.tensor([15,25,40,101])
```

---

# 7. Decoder Model Architecture

High level architecture:

```
Input Tokens

      |
      v

Embedding Layer

      |
      v

Positional Encoding

      |
      v

Transformer Decoder Blocks

      |
      v

Linear Projection Layer

      |
      v

Vocabulary Prediction

      |
      v

Generated Response
```

---

# 8. Embedding Layer

## Problem

Token IDs do not contain meaning.

Example:

```
dog = 10

cat = 200
```

The number difference does not represent meaning.

---

## Solution

Embedding converts tokens into vectors.

Example:

```
dog

   |

[0.34, 0.67, 0.21]
```

Similar words learn similar vector positions.

Example:

```
dog  ---- close ---- cat


dog  ---- far ------ airplane
```

---

# 9. Positional Encoding

Transformers process all tokens together.

They do not naturally understand word order.

Example:

Sentence 1:

```
Customer cancelled order
```

Sentence 2:

```
Order cancelled customer
```

Same words but different meaning.

Therefore:

```
Final Representation =
Word Embedding + Position Information
```

---

# 10. Self Attention

Attention helps the model understand relationships between words.

Example:

```
The order was delayed because it was shipped late.
```

Question:

What does "it" refer to?

Attention learns:

```
it
 |
 |
order
```

---

# 11. Transformer Decoder

Decoder generates text token by token.

Example:

Input:

```
refund_status ORD123
```

Generation:

Step 1:

```
I
```

Step 2:

```
I can
```

Step 3:

```
I can help
```

Step 4:

```
I can help with refund
```

This process is called:

# Autoregressive Generation

Meaning:

Previous generated tokens influence the next token.

---

# 12. Training Process

Initially model generates random output.

Example:

```
banana delivery hello
```

Expected:

```
Your refund is processing
```

Difference is calculated.

The model updates its weights.

Training cycle:

```
Forward Pass

      |
      v

Prediction

      |
      v

Calculate Loss

      |
      v

Backpropagation

      |
      v

Optimizer Updates Weights
```

---

# 13. Loss Function

We use:

# Cross Entropy Loss

Purpose:

Measure prediction error.

Example:

Correct word:

```
refund
```

Prediction:

```
order   -> 70%

refund  -> 20%

cancel  -> 10%
```

Loss pushes model to increase probability of:

```
refund
```

---

# 14. Optimizer

Optimizer used:

# Adam

Full form:

Adaptive Moment Estimation

Purpose:

Adjust neural network parameters to reduce loss.

---

# 15. Output File

The notebook generates:

```
outputs/

    decoder_predictions.json
```

Example:

```json
{
    "input":
    "intent order_status order ORD841324",

    "expected":
    "I can help with order ORD841324",

    "generated":
    "I can help with order ORD387515"
}
```

---

# 16. Result Analysis

## Successful Scenario

Input:

```
product_availability
```

Expected:

```
I can help with product availability.
Please share missing details.
```

Generated:

```
I can help with product availability.
Please share missing details.
```

Result:

SUCCESS

---

# 17. Observed Limitation

Input:

```
order_id : ORD841324
```

Expected:

```
ORD841324
```

Generated:

```
ORD387515
```

Issue:

Slot value copying failure.

---

# Why did this happen?

Because:

- Dataset is small
- IDs are unique
- Model learns patterns
- Model does not have explicit copy mechanism

---

# 18. Industry Improvements

## Pointer Generator Network

Allows model to copy values directly.

Example:

```
Input:

ORD123

 |
 |
Copy Mechanism

 |
 |

Output:

ORD123
```

---

## Pretrained LLM

Large Language Models already understand instructions:

Example:

```
Always preserve order IDs exactly.
```

---

# 19. Decoder vs LLM Comparison

| Feature | Decoder Transformer | LLM |
|---|---|---|
| Training | Required | Already trained |
| Dataset Need | High | Low |
| Speed | Fast | Slower |
| Cost | Low | Higher |
| Language Quality | Medium | High |
| Slot Copying | Weak | Strong |

---

# Final Summary

The Decoder Transformer successfully learns:

- Domain language
- Intent based generation
- Response templates

However, it struggles with unseen slot values.

This motivates comparing it with modern LLM based Controlled NLG.

```
Decoder Transformer
        |
        |
Understand limitations
        |
        |
Compare with LLM
```
