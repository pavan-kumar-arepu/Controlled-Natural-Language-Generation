<<<<<<< HEAD
# Controlled Natural Language Generation (NLG)
## Decoder Transformer vs Large Language Model (LLM)

## Problem Statement 2

### Title

**Controlled Natural Language Generation — Decoder vs LLM for Slot-Conditioned Response Generation**

---

# 1. Project Objective

The objective of this project is to design, implement, and compare two Natural Language Generation (NLG) approaches for generating customer service responses from structured inputs.

Two approaches are implemented:

1. Custom Transformer Decoder Model
2. Pretrained Large Language Model (LLM)

The goal is to analyze:

- Response quality
- Slot value preservation
- Generalization ability
- Model limitations
- Deployment trade-offs

---

# 2. Problem Definition

Input is provided as:

- Intent
- Slot values

Example:

Input:

```text
intent : order_status

slots:

order_id : ORD841324

channel : website
```

Expected generated response:

```text
I can help with order status for order ORD841324.
```

The model must:

- Understand the intent
- Generate a natural response
- Preserve important slot values
- Avoid hallucination

---

# 3. Complete Solution Pipeline

```text

                 Dataset Creation

                         |
                         v

                 Dataset Validation

                         |
                         v

                 Slot Linearization

                         |
                         v

                 Vocabulary Creation

                         |
                         v

        +--------------------------------+

        |                                |

        v                                v


 Decoder Transformer                 LLM Generator


        |                                |


 Train Custom Model              Prompt Engineering


        |                                |


 Generate Response               Generate Response


        |                                |

        +--------------------------------+

                         |

                         v

                  Evaluation Metrics

                         |

                         v

                  Final Analysis

```

---

# 4. Project Structure

```text
AssignmentSolution/

├── data/
│
│   ├── schema.json
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   │
│   ├── linearized_train.json
│   ├── linearized_validation.json
│   ├── linearized_test.json
│   │
│   ├── vocab.json
│   ├── token_to_id.json
│   └── id_to_token.json
│

├── notebooks/
│
│   ├── 01_data_generation.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_linearization.ipynb
│   ├── 04_vocab_creation.ipynb
│   ├── 05_decoder_transformer.ipynb
│   ├── 06_llm_generation.ipynb
│   ├── 07_evaluation.ipynb
│   └── 08_analysis_report.ipynb
│

├── outputs/
│
│   ├── decoder_predictions.json
│   ├── llm_predictions.json
│   └── evaluation_results.csv
│

├── requirements.txt

└── README.md

```

---

# 5. Step 01 - Dataset Generation

## Objective

Create a synthetic customer service dataset.

Generated:

- Training dataset
- Validation dataset
- Test dataset


Example intents:

```text
order_status

refund_status

modify_order

product_availability
```

Example slots:

```text
order_id

refund_id

products

delivery_slot

payment_method
```

---

# 6. Step 02 - Dataset Validation


Validation checks:

- Dataset count
- Intent distribution
- Slot availability
- Missing values
- Data correctness


Purpose:

Ensure generated data is suitable for model training.

---

# 7. Step 03 - Slot Linearization


Transform structured data into sequence format.


Before:

```json
{
 "intent":"order_status",
 "order_id":"ORD123"
}
```


After:

```text
[BOS]

intent : order_status

<sep>

order_id : ORD123

[EOS]
```


Why:

Transformer models process sequential input.

---

# 8. Step 04 - Vocabulary Creation


Purpose:

Convert text tokens into numerical IDs.


Example:

Vocabulary:

```json
{
 "[PAD]" : 0,

 "[BOS]" : 1,

 "order_status" : 25
}
```


Conversion:

```text
order_status

↓

25
```


Generated:

```text
vocab.json

token_to_id.json

id_to_token.json
```

---

# 9. Step 05 - Custom Decoder Transformer


## Architecture


```text

Input Tokens

      |

      v

Embedding Layer

      |

      v

Positional Encoding

      |

      v

Transformer Decoder

      |

      v

Linear Layer

      |

      v

Generated Response

```


Implemented using:

- PyTorch
- Transformer architecture


Concepts:

- Token Embedding
- Positional Encoding
- Self Attention
- Decoder blocks
- Autoregressive generation


Training:

- Loss Function:
  Cross Entropy Loss

- Optimizer:
  Adam Optimizer


Output:

```text
outputs/decoder_predictions.json
```

---

## Decoder Observation


Example:

Expected:

```text
I can help with order status for order ORD841324.
```


Generated:

```text
I can help with order status for order ORD387515.
```


Observation:

The decoder learned response format but struggled with unseen slot values.

Reason:

- Limited training examples
- No copy mechanism

---

# 10. Step 06 - LLM Based Generation


Model:

```text
FLAN-T5
```


Approach:

```text

Intent + Slots

      |

Input Preprocessing

      |

Few Shot Prompt Engineering

      |

Pretrained LLM

      |

Generated Response

```


Advantages:

- No training required
- Better generalization
- Better slot copying


Output:

```text
outputs/llm_predictions.json
```

---

# 11. Step 07 - Evaluation


Models evaluated using:


## BLEU Score

Full Form:

Bilingual Evaluation Understudy


Purpose:

Measures word overlap between generated and reference responses.


---


## Slot Accuracy


Purpose:

Measures whether important slot values are preserved.


Example:

Expected:

```text
ORD841324
```

Generated:

```text
ORD841324
```

Correct.


---


## Exact Match


Purpose:

Checks complete sentence match.


---

# 12. Evaluation Results


| Metric | Decoder Transformer | FLAN-T5 LLM |
|---|---:|---:|
| BLEU Score | 0.597 | 0.110 |
| Slot Accuracy | 0.267 | 0.867 |
| Exact Match | 0.067 | 0.000 |


---

# 13. Result Analysis


## Decoder Transformer


Strengths:

- Higher BLEU score
- Learns domain templates
- Lightweight model


Limitations:

- Poor unseen slot copying
- Requires training data
- Limited generalization



---

## LLM FLAN-T5


Strengths:

- Higher slot accuracy
- Handles unseen values better
- No custom training required


Limitations:

- Lower BLEU due to wording differences
- Prompt sensitive
- Larger compute requirement


---

# 14. Final Conclusion


The experiment demonstrates that both approaches have different advantages.


The Decoder Transformer achieved better lexical similarity because it learned dataset response patterns.


However, the LLM achieved significantly better slot preservation.


For real-world conversational AI systems:

- Decoder models are suitable for fixed domains with limited resources.

- LLM based systems are preferred where flexibility and correctness of dynamic values are important.


A hybrid production solution can combine:

```text

LLM Generation

+

Slot Validation Layer

+

Business Rules

```


to achieve high quality and reliable responses.

---

# 15. Setup Instructions


Create environment:

```bash
python -m venv venv
```


Activate:

Mac/Linux:

```bash
source venv/bin/activate
```


Install dependencies:

```bash
pip install -r requirements.txt
```


Run notebooks sequentially:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```


---

# 16. Technologies Used


- Python 3.11
- PyTorch
- HuggingFace Transformers
- FLAN-T5
- Pandas
- NLTK
- Jupyter Notebook
- VS Code


---

# End
=======
# Controlled Natural Language Generation (NLG)
## Decoder Transformer vs Large Language Model (LLM)

## Problem Statement 2

### Title

**Controlled Natural Language Generation — Decoder vs LLM for Slot-Conditioned Response Generation**

---

# 1. Project Objective

The objective of this project is to design, implement, and compare two Natural Language Generation (NLG) approaches for generating customer service responses from structured inputs.

Two approaches are implemented:

1. Custom Transformer Decoder Model
2. Pretrained Large Language Model (LLM)

The goal is to analyze:

- Response quality
- Slot value preservation
- Generalization ability
- Model limitations
- Deployment trade-offs

---

# 2. Problem Definition

Input is provided as:

- Intent
- Slot values

Example:

Input:

```text
intent : order_status

slots:

order_id : ORD841324

channel : website
```

Expected generated response:

```text
I can help with order status for order ORD841324.
```

The model must:

- Understand the intent
- Generate a natural response
- Preserve important slot values
- Avoid hallucination

---

# 3. Complete Solution Pipeline

```text

                 Dataset Creation

                         |
                         v

                 Dataset Validation

                         |
                         v

                 Slot Linearization

                         |
                         v

                 Vocabulary Creation

                         |
                         v

        +--------------------------------+

        |                                |

        v                                v


 Decoder Transformer                 LLM Generator


        |                                |


 Train Custom Model              Prompt Engineering


        |                                |


 Generate Response               Generate Response


        |                                |

        +--------------------------------+

                         |

                         v

                  Evaluation Metrics

                         |

                         v

                  Final Analysis

```

---

# 4. Project Structure

```text
AssignmentSolution/

├── data/
│
│   ├── schema.json
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   │
│   ├── linearized_train.json
│   ├── linearized_validation.json
│   ├── linearized_test.json
│   │
│   ├── vocab.json
│   ├── token_to_id.json
│   └── id_to_token.json
│

├── notebooks/
│
│   ├── 01_data_generation.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_linearization.ipynb
│   ├── 04_vocab_creation.ipynb
│   ├── 05_decoder_transformer.ipynb
│   ├── 06_llm_generation.ipynb
│   ├── 07_evaluation.ipynb
│   └── 08_analysis_report.ipynb
│

├── outputs/
│
│   ├── decoder_predictions.json
│   ├── llm_predictions.json
│   └── evaluation_results.csv
│

├── requirements.txt

└── README.md

```

---

# 5. Step 01 - Dataset Generation

## Objective

Create a synthetic customer service dataset.

Generated:

- Training dataset
- Validation dataset
- Test dataset


Example intents:

```text
order_status

refund_status

modify_order

product_availability
```

Example slots:

```text
order_id

refund_id

products

delivery_slot

payment_method
```

---

# 6. Step 02 - Dataset Validation


Validation checks:

- Dataset count
- Intent distribution
- Slot availability
- Missing values
- Data correctness


Purpose:

Ensure generated data is suitable for model training.

---

# 7. Step 03 - Slot Linearization


Transform structured data into sequence format.


Before:

```json
{
 "intent":"order_status",
 "order_id":"ORD123"
}
```


After:

```text
[BOS]

intent : order_status

<sep>

order_id : ORD123

[EOS]
```


Why:

Transformer models process sequential input.

---

# 8. Step 04 - Vocabulary Creation


Purpose:

Convert text tokens into numerical IDs.


Example:

Vocabulary:

```json
{
 "[PAD]" : 0,

 "[BOS]" : 1,

 "order_status" : 25
}
```


Conversion:

```text
order_status

↓

25
```


Generated:

```text
vocab.json

token_to_id.json

id_to_token.json
```

---

# 9. Step 05 - Custom Decoder Transformer


## Architecture


```text

Input Tokens

      |

      v

Embedding Layer

      |

      v

Positional Encoding

      |

      v

Transformer Decoder

      |

      v

Linear Layer

      |

      v

Generated Response

```


Implemented using:

- PyTorch
- Transformer architecture


Concepts:

- Token Embedding
- Positional Encoding
- Self Attention
- Decoder blocks
- Autoregressive generation


Training:

- Loss Function:
  Cross Entropy Loss

- Optimizer:
  Adam Optimizer


Output:

```text
outputs/decoder_predictions.json
```

---

## Decoder Observation


Example:

Expected:

```text
I can help with order status for order ORD841324.
```


Generated:

```text
I can help with order status for order ORD387515.
```


Observation:

The decoder learned response format but struggled with unseen slot values.

Reason:

- Limited training examples
- No copy mechanism

---

# 10. Step 06 - LLM Based Generation


Model:

```text
FLAN-T5
```


Approach:

```text

Intent + Slots

      |

Input Preprocessing

      |

Few Shot Prompt Engineering

      |

Pretrained LLM

      |

Generated Response

```


Advantages:

- No training required
- Better generalization
- Better slot copying


Output:

```text
outputs/llm_predictions.json
```

---

# 11. Step 07 - Evaluation


Models evaluated using:


## BLEU Score

Full Form:

Bilingual Evaluation Understudy


Purpose:

Measures word overlap between generated and reference responses.


---


## Slot Accuracy


Purpose:

Measures whether important slot values are preserved.


Example:

Expected:

```text
ORD841324
```

Generated:

```text
ORD841324
```

Correct.


---


## Exact Match


Purpose:

Checks complete sentence match.


---

# 12. Evaluation Results


| Metric | Decoder Transformer | FLAN-T5 LLM |
|---|---:|---:|
| BLEU Score | 0.597 | 0.110 |
| Slot Accuracy | 0.267 | 0.867 |
| Exact Match | 0.067 | 0.000 |


---

# 13. Result Analysis


## Decoder Transformer


Strengths:

- Higher BLEU score
- Learns domain templates
- Lightweight model


Limitations:

- Poor unseen slot copying
- Requires training data
- Limited generalization



---

## LLM FLAN-T5


Strengths:

- Higher slot accuracy
- Handles unseen values better
- No custom training required


Limitations:

- Lower BLEU due to wording differences
- Prompt sensitive
- Larger compute requirement


---

# 14. Final Conclusion


The experiment demonstrates that both approaches have different advantages.


The Decoder Transformer achieved better lexical similarity because it learned dataset response patterns.


However, the LLM achieved significantly better slot preservation.


For real-world conversational AI systems:

- Decoder models are suitable for fixed domains with limited resources.

- LLM based systems are preferred where flexibility and correctness of dynamic values are important.


A hybrid production solution can combine:

```text

LLM Generation

+

Slot Validation Layer

+

Business Rules

```


to achieve high quality and reliable responses.

---

# 15. Setup Instructions


Create environment:

```bash
python -m venv venv
```


Activate:

Mac/Linux:

```bash
source venv/bin/activate
```


Install dependencies:

```bash
pip install -r requirements.txt
```


Run notebooks sequentially:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```


---

# 16. Technologies Used


- Python 3.11
- PyTorch
- HuggingFace Transformers
- FLAN-T5
- Pandas
- NLTK
- Jupyter Notebook
- VS Code


---

# End
>>>>>>> f3d3738 (Add STEP 9 technical report and finalize ConversationAI assignment with all outputs)
