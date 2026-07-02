#!/usr/bin/env python3
"""
Launch FinePhrase runs from local JSONL files, saving each split locally before
uploading the whole tree to the Hugging Face Hub.

This is the local-input / local-output variant of ``finephrase.py``. It reads two
(or more) local JSONL files -- one per **subset** -- and, for each subset, runs
every prompt template (math / table / faq / tutorial) as a separate **split**.

Output layout (one ``python finephrase_local.py`` run):

    {LOCAL_OUTPUT_DIR}/{subset}/{split}/*.parquet

After all runs finish, the entire ``LOCAL_OUTPUT_DIR`` tree is uploaded to a single
HF dataset repo, preserving the ``{subset}/{split}/`` structure.

This script uses the best configuration found from the two-tier benchmark optimization:
   tp=1, mns=2048, mnbt=32768, gmu=0.95, spec=suffix_32 (1.75x speedup)

Usage:
    # Fill in INPUT_SUBSETS and LOCAL_OUTPUT_DIR below, then:
    python examples/inference/finephrase_local.py
"""

from typing import Any

from generate_data import main as generate_data_main
from huggingface_hub import upload_large_folder
from utils import check_hf_auth, ensure_repo_exists, resolve_repo_id

from datatrove.utils.logging import logger


# nohup python -u examples/inference/finephrase_local.py > finephrase_local.log 2>&1 &
INPUT_SUBSETS: dict[str, str] = {
    "gigaverbo2": "/raid/aluno_edward/raw_medium_data_pt/gigaverbo2_filtered.jsonl",
    "pt_fineweb2": "/raid/aluno_edward/raw_medium_data_pt/pt_fineweb2_filtered.jsonl",
}

# Base local directory for the parquet output tree (later uploaded to HF).
LOCAL_OUTPUT_DIR = "finephrase_pretrain_p1"


KWARGS: dict[str, Any] = {
    "time": "7-00:00:00",
    "qos": "low",
    "model_name_or_path": "google/gemma-4-26B-A4B-it",
    "model_max_context": 16000,
    "max_tokens": 4096,  # Could do 4096, but the model often does not generate as much anyway
    "prompt_column": "text",
    "output_dataset_name": "EdwardSJ151/sites_educacionais_rephrase",
    "output_dir": "sites_educacionais_rephrase",
    "output_private": True,
    "max_concurrent_generations": 5000,
    "max_concurrent_documents": 5000,
    "max_num_seqs": 2048,
    "max_num_batched_tokens": 32768,  # Optimized for 32768 tokens, but we use 16384 tokens to avoid OOM
    "gpu_memory_utilization": 0.90,  # Optimized one is 0.95, but this leads to OOM sometimes
    "speculative_config": '{"method": "suffix", "num_speculative_tokens": 32}',
    "enable_monitoring": False,  # ignored in local-output mode
    "examples_per_chunk": 100_000,  # 500 are around 2MB => 100_000 are around 400MB (~270MB compressed)
    "optimization_level": 3,  # Set to 0 for debugging
    "local_execution": True,
    "workers": 1,
    "tasks": 1,
    "tp": 1,
    "dp": 2,
    # Local output: write parquet to disk (skips per-run HF upload/datacard).
    "local_output_path": LOCAL_OUTPUT_DIR,
}

PROMPT_TEMPLATES_ENGLISH: dict[str, str] = {
    "math": (
        "Rewrite the document to create a mathematical word problem based on the numerical data or relationships in "
        "the text. Provide a step-by-step solution that shows the calculation process clearly. Create a problem that "
        "requires multi-step reasoning and basic arithmetic operations. It should include the question followed by a "
        "detailed solution showing each calculation step. Output only the problem and solution, nothing else.\n\n"
        "Document: [[DOCUMENT]]"
    ),
    "table": (
        "Rewrite the document as a structured table that organizes the key information, then generate one "
        "question-answer pair based on the table. First extract the main data points and organize them into a clear "
        "table format with appropriate headers using markdown table syntax with proper alignment. After the table, "
        "generate one insightful question that can be answered using the table data. Provide a clear, concise answer "
        "to the question based on the information in the table. Output only the table followed by the question-answer "
        "pair, nothing else.\n\nDocument: [[DOCUMENT]]"
    ),
    "faq": (
        "Rewrite the document as a comprehensive FAQ (Frequently Asked Questions). Extract or infer the key questions "
        "a reader would have about this topic, then provide clear, direct answers. Order questions logically, from "
        "foundational to advanced, or by topic area. Each answer should be self-contained and understandable without "
        "reference to other answers. Ensure the FAQ works as a standalone document. Output only the FAQ, nothing "
        "else.\n\nDocument: [[DOCUMENT]]"
    ),
    "tutorial": (
        "Rewrite the document as a clear, step-by-step tutorial or instructional guide. Use numbered steps or bullet "
        "points where appropriate to enhance clarity. Preserve all essential information while ensuring the style "
        "feels didactic and easy to follow. Output only the tutorial, nothing else.\n\nDocument: [[DOCUMENT]]"
    ),
}


# Default rephrasing prompts for Brazilian educational content (set `PROMPT_TEMPLATES = PROMPT_TEMPLATES_ENGLISH` to use EN).
PROMPT_TEMPLATES: dict[str, str] = {
    "math": (
        "Reescreva o documento em português para criar um problema matemático baseado nos dados numéricos ou nas "
        "relações presentes no texto. Forneça uma solução passo a passo que mostre claramente o processo de cálculo. "
        "Crie um problema que exija raciocínio em múltiplas etapas e operações aritméticas básicas. Deve incluir a "
        "pergunta seguida de uma solução detalhada mostrando cada etapa do cálculo. Produza apenas o problema e a "
        "solução, nada mais.\n\n"
        "Documento: [[DOCUMENT]]"
    ),
    "table": (
        "Reescreva o documento em português como uma tabela estruturada que organize as principais informações, e em "
        "seguida gere um par de pergunta e resposta baseado na tabela. Primeiro extraia os dados principais e organize-os "
        "em um formato de tabela claro com cabeçalhos apropriados usando a sintaxe de tabela em markdown com alinhamento "
        "correto. Após a tabela, gere uma pergunta relevante que possa ser respondida usando os dados da tabela. Forneça "
        "uma resposta clara e concisa baseada nas informações da tabela. Produza apenas a tabela seguida do par de "
        "pergunta e resposta, nada mais.\n\nDocumento: [[DOCUMENT]]"
    ),
    "faq": (
        "Reescreva o documento em português como um FAQ abrangente (Perguntas Frequentes). Extraia ou infira as "
        "principais perguntas que um leitor teria sobre este tema, e forneça respostas claras e diretas. Organize as "
        "perguntas logicamente, do básico ao avançado ou por áreas temáticas. Cada resposta deve ser autossuficiente "
        "e compreensível sem referência a outras respostas. Garanta que o FAQ funcione como um documento independente. "
        "Produza apenas o FAQ, nada mais.\n\nDocumento: [[DOCUMENT]]"
    ),
    "tutorial": (
        "Reescreva o documento em português como um tutorial claro, passo a passo, ou guia instrucional. Use etapas "
        "numeradas ou marcadores quando apropriado para melhorar a clareza. Preserve todas as informações essenciais "
        "enquanto garante que o estilo seja didático e fácil de seguir. Produza apenas o tutorial, nada mais.\n\n"
        "Documento: [[DOCUMENT]]"
    ),
}


def launch_template(subset: str, input_path: str, template_name: str, template_text: str) -> None:
    """Run one FinePhrase generation for a (subset, prompt template) pair.

    Args:
        subset: Subset name (top-level output folder / HF config).
        input_path: Local JSONL file to read for this subset.
        template_name: Prompt template identifier (used as the split name).
        template_text: Prompt template containing the ``[[DOCUMENT]]`` placeholder.
    """
    run_name = f"finephrase_{subset}_{template_name}"
    logger.info(f"Launching FinePhrase run '{run_name}' (input: {input_path})")
    generate_data_main(
        **KWARGS,
        name=run_name,
        prompt_template=[template_name, template_text],
        input_local_path=input_path,
        output_subset=subset,
        # Scope run metadata (logs/checkpoints/completions) per subset so runs of the
        # same template across subsets don't collide and get skipped as "already completed".
        output_dir=f"{KWARGS['output_dir']}/{subset}",
    )


def main() -> None:
    """Run every subset x template locally, then upload the tree to the HF Hub."""
    # Fail fast on auth/repo before hours of inference.
    check_hf_auth()
    full_repo_id = resolve_repo_id(KWARGS["output_dataset_name"])
    ensure_repo_exists(full_repo_id, private=KWARGS["output_private"])

    num_runs = 0
    for subset, input_path in INPUT_SUBSETS.items():
        for template_name, template_text in PROMPT_TEMPLATES.items():
            launch_template(subset, input_path, template_name, template_text)
            num_runs += 1

    logger.info(
        f"Completed {num_runs} FinePhrase runs "
        f"({len(INPUT_SUBSETS)} subsets x {len(PROMPT_TEMPLATES)} splits). "
        f"Local output at '{LOCAL_OUTPUT_DIR}'."
    )

    logger.info(f"Uploading '{LOCAL_OUTPUT_DIR}' to HF dataset repo '{full_repo_id}'")
    upload_large_folder(
        repo_id=full_repo_id,
        repo_type="dataset",
        folder_path=LOCAL_OUTPUT_DIR,
    )
    logger.info(f"Upload to '{full_repo_id}' complete.")


if __name__ == "__main__":
    main()
