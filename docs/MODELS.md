# LLM Models for Logical Reasoning Experiments

This document lists the models available for testing on FOLIO and Multi-LogiEval benchmarks via OpenRouter.

## Latest Models (December 2025)

### Tier 1: Reasoning-Focused Models (Recommended)

| Model | OpenRouter ID | Notes |
|-------|---------------|-------|
| **DeepSeek-R1-0528** | `deepseek/deepseek-r1-0528` | Best open reasoning model, 87.5% AIME |
| **DeepSeek-V3.2** | `deepseek/deepseek-v3.2` | Latest V3, GPT-5 class performance |
| **Qwen3-235B** | `qwen/qwen3-235b-a22b` | Strong reasoning, 235B MoE |
| **Qwen3-Max** | `qwen/qwen3-max` | Alibaba's flagship |
| **Mistral Large 2512** | `mistralai/mistral-large-2512` | Latest Mistral Large (Dec 2025) |

### Tier 2: Frontier Closed Models

| Model | OpenRouter ID | Notes |
|-------|---------------|-------|
| **Claude Opus 4.5** | `anthropic/claude-opus-4.5` | Anthropic's best |
| **Claude Sonnet 4.5** | `anthropic/claude-sonnet-4.5` | Fast + capable |
| **Gemini 2.5 Pro** | `google/gemini-2.5-pro` | 1M context, Deep Think |
| **Grok 4** | `x-ai/grok-4` | xAI's latest |

### Tier 3: Open-Source Models

| Model | OpenRouter ID | Notes |
|-------|---------------|-------|
| **Llama 4 Maverick** | `meta-llama/llama-4-maverick` | 400B MoE, 256K context |
| **Llama 4 Scout** | `meta-llama/llama-4-scout` | 10M context window |
| **Llama 3.3 70B** | `meta-llama/llama-3.3-70b-instruct` | Strong baseline |
| **QwQ 32B** | `qwen/qwq-32b` | Reasoning specialist |
| **Qwen3-Next 80B** | `qwen/qwen3-next-80b-a3b-thinking` | Thinking mode |

### All Available Models by Provider

#### DeepSeek
```
deepseek/deepseek-r1-0528          # Best reasoning (R1 May update)
deepseek/deepseek-r1               # R1 base
deepseek/deepseek-v3.2             # Latest V3 (GPT-5 class)
deepseek/deepseek-v3.2-exp         # Experimental
deepseek/deepseek-chat-v3.1        # Chat optimized
deepseek/deepseek-prover-v2        # Math proofs
```

#### Mistral
```
mistralai/mistral-large-2512       # Latest Large (Dec 2025)
mistralai/mistral-large-2411       # November Large
mistralai/mistral-medium-3.1       # Medium tier
mistralai/mistral-small-3.2-24b-instruct  # Small 3.2
mistralai/ministral-14b-2512       # 14B (edge)
mistralai/ministral-8b-2512        # 8B (edge)
mistralai/ministral-3b-2512        # 3B (edge)
mistralai/pixtral-large-2411       # Multimodal
mistralai/codestral-2508           # Code specialist
```

#### Qwen
```
qwen/qwen3-235b-a22b               # 235B MoE flagship
qwen/qwen3-235b-a22b-2507          # July version
qwen/qwen3-max                     # Alibaba's best
qwen/qwen3-next-80b-a3b-thinking   # Thinking mode
qwen/qwen3-next-80b-a3b-instruct   # Instruct mode
qwen/qwen3-coder                   # Code specialist
qwen/qwen3-32b                     # Dense 32B
qwen/qwq-32b                       # Reasoning specialist
qwen/qwen-max                      # Qwen2 Max
```

#### Meta Llama
```
meta-llama/llama-4-maverick        # 400B MoE, 256K context
meta-llama/llama-4-scout           # 10M context window
meta-llama/llama-3.3-70b-instruct  # Strong 70B
meta-llama/llama-3.1-405b-instruct # Largest Llama 3
```

#### Anthropic Claude
```
anthropic/claude-opus-4.5          # Best Claude
anthropic/claude-opus-4.1          # Opus 4.1
anthropic/claude-opus-4            # Opus 4
anthropic/claude-sonnet-4.5        # Fast Sonnet
anthropic/claude-sonnet-4          # Sonnet 4
anthropic/claude-3.7-sonnet        # 3.7 Sonnet
anthropic/claude-haiku-4.5         # Fastest
```

#### Google Gemini
```
google/gemini-2.5-pro              # 1M context, Deep Think
google/gemini-2.5-flash            # Fast
google/gemini-3-pro-preview        # Preview
```

#### xAI Grok
```
x-ai/grok-4                        # Latest Grok
x-ai/grok-4-fast                   # Fast version
x-ai/grok-4.1-fast                 # 4.1 Fast
x-ai/grok-3                        # Grok 3
x-ai/grok-3-mini                   # Mini version
```

## Usage Examples

### Environment Setup
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Running Experiments

```bash
# DeepSeek-R1-0528 (best open reasoning)
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_folio_async.py \
    --api_key $OPENROUTER_API_KEY \
    --model "deepseek/deepseek-r1-0528" \
    --folio_file data/folio_original/folio-validation.json \
    --system_prompt prompts/folio/cot_system.txt \
    --user_prompt prompts/folio/cot_user.txt \
    --num_stories 0 --concurrency 5 --prompt_name deepseek_r1_cot

# Mistral Large 2512 (latest)
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_folio_async.py \
    --api_key $OPENROUTER_API_KEY \
    --model "mistralai/mistral-large-2512" \
    --folio_file data/folio_original/folio-validation.json \
    --system_prompt prompts/folio/cot_system.txt \
    --user_prompt prompts/folio/cot_user.txt \
    --num_stories 0 --concurrency 5 --prompt_name mistral_large_cot

# Qwen3-235B
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_folio_async.py \
    --api_key $OPENROUTER_API_KEY \
    --model "qwen/qwen3-235b-a22b" \
    --folio_file data/folio_original/folio-validation.json \
    --system_prompt prompts/folio/cot_system.txt \
    --user_prompt prompts/folio/cot_user.txt \
    --num_stories 0 --concurrency 5 --prompt_name qwen3_235b_cot

# Llama 4 Maverick
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_folio_async.py \
    --api_key $OPENROUTER_API_KEY \
    --model "meta-llama/llama-4-maverick" \
    --folio_file data/folio_original/folio-validation.json \
    --system_prompt prompts/folio/cot_system.txt \
    --user_prompt prompts/folio/cot_user.txt \
    --num_stories 0 --concurrency 5 --prompt_name llama4_maverick_cot
```

### Short Aliases
You can also use short aliases in the code:
```python
# These resolve automatically
"deepseek-r1"     -> "deepseek/deepseek-r1-0528"
"deepseek-v3.2"   -> "deepseek/deepseek-v3.2"
"mistral-large"   -> "mistralai/mistral-large-2512"
"qwen3-235b"      -> "qwen/qwen3-235b-a22b"
"llama4-maverick" -> "meta-llama/llama-4-maverick"
"claude-opus"     -> "anthropic/claude-opus-4.5"
"claude-sonnet"   -> "anthropic/claude-sonnet-4.5"
"gemini-pro"      -> "google/gemini-2.5-pro"
"grok4"           -> "x-ai/grok-4"
```

## Model Comparison Results

| Dataset | Model | Accuracy |
|---------|-------|----------|
| FOLIO | GPT-5 Bidirectional | 87.68% |
| FOLIO | GPT-5 CoT (default) | 85.71% |
| Multi-LogiEval | GPT-5 Bidirectional | 82.67% |

*Results for open-source models TBD*
