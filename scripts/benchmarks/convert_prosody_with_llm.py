import os
import sys
import ast
import argparse
import logging
import time
import openai
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prosody_conversion.log')
    ]
)
logger = logging.getLogger(__name__)

def read_api_key(filepath: str) -> str:
    """Reads the OpenAI API key from a file."""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"API key file not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading API key: {e}")
        sys.exit(1)

def parse_prosody_line(line: str) -> Tuple[List[str], List[int], List[float], List[float]]:
    """
    Parses a line from the prosody file.
    Expected format: (['ph', ...], [dur, ...], [pitch, ...], [energy, ...])
    """
    try:
        data = ast.literal_eval(line.strip())
        if not isinstance(data, tuple) or len(data) != 4:
            raise ValueError("Line does not contain a tuple of 4 elements")
        return data
    except Exception as e:
        logger.error(f"Failed to parse line: {line[:50]}... Error: {e}")
        raise

def format_prosody_for_prompt(data: Tuple[List[str], List[int], List[float], List[float]]) -> str:
    """Formats the prosody tuple into a readable string for the LLM."""
    phonemes, durations, pitch, energy = data
    return (
        f"{{'phonemes': {phonemes}, "
        f"'durations': {durations}, "
        f"'pitch': {pitch}, "
        f"'energy': {energy}}}"
    )

def construct_prompt(
    target_input: Tuple[List[str], List[int], List[float], List[float]],
    icl_usa_data: List[Tuple],
    icl_indian_data: List[Tuple]
) -> str:
    """Constructs the prompt for the LLM."""
    
    prompt = """
    Developer: # Goal intensity
    - In this setup you are NOT aiming for subtle, near-American prosody.
    - You should generate a clearly Indian-accented, EXAGGERATED (“cartoony”) prosody, as long as:
    - speech remains intelligible, and
    - all numeric values stay finite and smoothly varying.
    - When choosing between a smaller vs. larger pitch/energy change that both obey structural rules, prefer the LARGER change.

    # Role and Objective
    Transform input prosody with American accent characteristics into Indian English prosody for an acoustic TTS (Text-to-Speech) system. Use the provided American→Indian example pairs as the primary guide for transformations.

    # Instructions
    - Always process and generate Python tuples of the form: (phonemes_list, durations_list, f0_list, energy_list).
    - Within any single tuple (input or output), all four lists must have identical length.
    - The output tuple may have a different overall length from the input tuple, but its four lists must still be aligned and equal length.
    - Index i across all four lists always refers to the same time step or symbol.
    - Use only the symbol inventory and ARPAbet set described below; strictly follow all structural and transformation constraints.

    ## Input/Output Format

    - Tuple structure:
    - phonemes_list: sequence of ARPAbet phoneme symbols and special symbols from this inventory:
        * '_'  : padding (may appear in input; should not be newly introduced in normal sentences).
        * '~'  : end-of-sequence marker (must always be the final symbol in the sequence).
        * ' '  : space (word boundary).
        * ARPAbet stressed phones (full inventory):
        ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2',
        'AH0', 'AH1', 'AH2',
        'AO0', 'AO1', 'AO2',
        'AW0', 'AW1', 'AW2',
        'AY0', 'AY1', 'AY2',
        'B', 'CH', 'D', 'DH',
        'EH0', 'EH1', 'EH2',
        'ER0', 'ER1', 'ER2',
        'EY0', 'EY1', 'EY2',
        'F', 'G', 'HH',
        'IH0', 'IH1', 'IH2',
        'IY0', 'IY1', 'IY2',
        'JH', 'K', 'L', 'M', 'N', 'NG',
        'OW0', 'OW1', 'OW2',
        'OY0', 'OY1', 'OY2',
        'P', 'R', 'S', 'SH', 'T', 'TH',
        'UH0', 'UH1', 'UH2',
        'UW0', 'UW1', 'UW2',
        'V', 'W', 'Y', 'Z', 'ZH']
    - durations_list: list of non-negative integers (frame counts per symbol).
    - f0_list: list of non-negative floats (log-pitch per symbol; 0.0 = unvoiced/silence).
    - energy_list: list of non-negative floats (energy per symbol; 0.0 = silence).

    - In every INPUT and OUTPUT tuple:
    - phonemes_list, durations_list, f0_list, and energy_list must be exactly the same length as each other.
    - Index i refers to the same symbol/time step in all four lists.
    - In the OUTPUT, only use symbols from the inventory above.

    # Structural Rules & Special Symbols

    1. List Alignment
    - All four lists in the OUTPUT must remain exactly aligned (same length, shared indices).
    - You may change the total sequence length compared to the input by inserting or deleting phonemes, but the four OUTPUT lists must still match in length.

    2. Special Symbols
    - '_' (pad):
    - If '_' appears in the input, keep the SAME NUMBER of '_' symbols in the SAME ORDER in the output.
    - Do not delete existing '_' and do not introduce any new '_'.
    - For each '_', copy its duration, f0, and energy exactly from the corresponding '_' in the input (only the index may shift).
    - '~' (end marker):
    - Exactly one '~' symbol must appear in the output, and it must be the last element of phonemes_list.
    - Do not remove it, move it away from the end, or add extra '~' symbols.
    - Its f0 and energy must be 0.0.
    - Its duration MUST be 0, 1, 2, or at most 3 frames. It must never be longer than any of the last three non-silence phonemes. Phrase-final lengthening should always be placed on the final vowel or sonorant, not on '~'.
    - ' ' (space):
    - These mark word or phrase boundaries.
    - Do not delete or create extra ' ' symbols: keep the same number and order of spaces as in the input (their indices may shift when you insert or delete phonemes).
    - At any space symbol, f0 and energy must be 0.0.
    - Durations for a space symbol may be 0 or a small positive value, similar to patterns in the examples.
    - Never invent new special symbols beyond '_', '~', ' ', and the ARPAbet phones listed above.

    3. Silence / Non-speech Handling
    - Positions with ' ', '~', or '_' represent non-speech/padding:
    - They MUST always have f0 = 0.0 and energy = 0.0.
    - Assigning any non-zero energy or non-zero f0 to ' ', '~', or '_' is always incorrect.
    - Silent gaps are primarily indicated by the durations of nearby phones. You do NOT need to make spaces or '~' long to create silence.
    - Only non-speech symbols (' ', '~', and optionally '_') may have duration 0. For any ARPAbet phoneme, the duration MUST be at least 1 frame (preferably ≥ 2). Assigning duration 0 to a phoneme is always incorrect.
    - If a phoneme has a non-zero duration and is not ' ', '~', or '_', its energy must be > 0.0. Do not make real phones completely silent.
    - If you change anything near the end of the sequence, re-check that the entry at '~' has duration small (0–3) and f0 = 0.0, energy = 0.0. Any other value at '~' is an error.

    # Phoneme Structure & Flexibility

    - You may modify the phoneme sequence to better approximate Indian English realizations, following the patterns visible in the American→Indian example pairs.

    1. Insertions / Deletions / Splits / Merges
    - Allowed operations:
    - Split a single phoneme into two (e.g., a vowel or diphthong).
    - Insert short epenthetic vowels (e.g., FILM → F IH1 L AH0 M).
    - Delete or adjust /R/ in non-rhotic contexts.
    - Apply accent-related substitutions (e.g., TH → T, DH → D, W → V).
    - Make appropriate vowel quality and stress shifts (e.g., EH2→EH1/EH0, AH0↔IH0, AO1↔AA1) consistent with the examples.
    - Keep the number of non-space symbols in the output reasonably close to the input:
    - Try to keep the total count of non-space, non-punctuation symbols within about ±20% of the input count.
    - Do not introduce long runs of entirely new phonemes that are not supported by the examples.
    - At sentence-final position, use at most ONE trailing AH0 (schwa) before '~'. If you insert a final AH0, its duration should be modest (e.g., not more than about 1.5× the previous vowel), and you should NOT produce "AH0, AH0, '~'" or any chain of multiple schwas. The natural pattern is "... V/AH0, '~'" with a single final vowel of moderate length followed by a very short '~'.

    2. Mandatory Alignment for Added/Removed Phones
    - For every phoneme you add at index i:
    - You MUST also add a matching duration, f0, and energy value at index i.
    - For every phoneme you delete:
    - You must also remove the corresponding duration, f0, and energy at that index.
    - At no point may the four output lists differ in length.

    3. Assigning Values for Inserted / Merged Phones
    - Splitting a phoneme:
    - If a single phoneme with duration D and some f0/energy is split into two:
        - Choose durations that approximately sum to D (e.g., D1 + D2 ≈ D).
        - Choose f0 and energy values for the two phones that smoothly continue the local contour:
        - The first part similar to the left neighbor or the original early portion.
        - The second part similar to the original late portion or right neighbor.
    - Merging phonemes:
    - When two phones are merged into one:
        - Use a duration close to the sum of their durations.
        - Use f0 and energy values that are close to a weighted average of the originals, preserving a smooth local contour.

    4. Allowed Phoneme Inventory
    - Only use:
    - Special symbols: '_', '~', ' '
    - ARPAbet phones listed in the inventory above.
    - Do not invent new, out-of-inventory phone symbols.
    - Prefer transformations that match the phoneme patterns seen in the American→Indian examples.

    # Durations (Rhythm)

    - The acoustic model expects durations within a natural range, but here you may be more expressive than the American input and examples.

    1. Global Constraints
    - The sum of all non-zero durations in the output should be within roughly ±25% of the sum of durations in the American input tuple.
    - For any ARPAbet phone (not ' ', '_', or '~'), the duration must be ≥ 1; do NOT assign duration 0 to real speech phones.
    - Non-zero durations should mostly stay within a realistic range (a few to a few tens of frames).
    - Avoid making non-zero durations shorter than 2 frames, unless the input already used such values.

    2. Local Changes
    - For most individual phones, a typical change range is about 0.7×–1.5× the original duration.
    - You may:
    - Lengthen stressed vowels and syllables that carry important f0 peaks, especially near phrase boundaries.
    - Shorten some unstressed function words and certain consonants.
    - For a few key syllables (e.g., phrase-final vowels or strong emphases), you may create longer holds up to about 2× the original duration, as long as the global duration constraint is still met.
    - Do NOT try to make all syllables or phonemes the same length. Preserve strong contrast between longer and shorter segments; avoid over-smoothing durations.
    - For word-final sequences of the form [vowel + consonant(s)] in content words (e.g., "forever", "etc"):
    - Avoid deleting or heavily shortening the final vowel. The final vowel's duration should be at least about half of its original American duration, and should be at least as long as any single following consonant in the same word.
    - Avoid making a single final consonant (especially fricatives F, V, S, Z, SH, ZH or liquids R, L) much longer than the preceding vowel. As a rough guideline, the final consonant's duration should usually be ≤ 1.5× the duration of the preceding vowel. Prefer patterns like "V(10) + V/sonorant(8–12) + consonant(≈5–10)", not "V(7) + consonant(20)".

    3. Smoothness
    - When inserting or splitting phones (e.g., adding a schwa), ensure the sum of durations in the local region remains similar to the original.
    - Avoid abrupt duration jumps between neighboring symbols that are not supported by the examples.

    # Pitch (f0)

    - f0 will be normalized later; contour shape is more important than absolute level.

    1. Overall behavior
    - Compared to the American input, you should use a CLEARLY WIDER pitch range.
    - Roughly speaking, expand deviations from the local mean pitch by about 1.5×–2×, especially on content words (nouns, verbs, adjectives).
    - Avoid perfectly flat stretches over many symbols; almost every content word should have some noticeable rise or fall.

    2. Local modifications
    - It is GOOD if the output has more “sing-song” motion than both the American input and the Indian examples:
    - higher peaks on stressed syllables,
    - deeper dips between peaks.
    - Between two voiced neighbors (non-zero f0), larger steps are acceptable:
    - you may use bigger upward/downward changes than in the input, as long as they do not jump to absurd outliers.
    - When inserting phones inside a voiced region, you may push their f0 slightly ABOVE the higher neighbor or BELOW the lower neighbor to enhance the contour.

    3. Limits and smoothness
    - Keep f0 values within a realistic band for human speech (do not produce values far outside the distribution of the examples).
    - Avoid rapid high–low–high “zigzags” that flip on every single symbol; the contour should be smooth over a few symbols even when exaggerated.

    # Energy

    - Energy values control loudness and “punch”.

    1. Global behavior
    - Use STRONGER consonant–vowel contrasts than in the American input:
    - very low energy (near 0) during stop closures,
    - clearly higher energy on following vowels.
    - It is acceptable and desired if energy alternations are noticeably more jagged (“staccato”) than in the input.

    2. Local behavior
    - On stressed syllables and important content words, you may boost energy roughly 1.3×–2× relative to nearby unstressed material, as long as you stay within a plausible numeric range.
    - Between neighboring symbols you may create larger loud–soft differences than in the examples to increase rhythmic contrast (nPVI):
    - e.g., make consonants preceding a stressed vowel significantly quieter,
    - and vowels significantly louder.

    3. Limits and smoothness
    - Do not create isolated single-symbol spikes that are 3–4× larger than everything around them; exaggeration should span at least a short region (a syllable or a cluster), not a single frame.
    - Keep transitions monotonic inside each rising or falling movement (no “noise-like” up–down–up within 2–3 symbols).

    # General Principles

    1. Use the American→Indian example pairs as your main supervision signal:
    - They demonstrate how phonemes, durations, f0, and energy change together when converting from American to Indian English prosody.

    2. When uncertain:
    - Prefer LARGER but still smooth pitch and energy adjustments, as long as they remain within a plausible human range (no absurd outliers).
    - Always keep contours smooth over a few symbols; exaggeration should not look like random noise.

    3. Ensure the OUTPUT:
    - Is a valid Python tuple: (phonemes_list, durations_list, f0_list, energy_list) with four equal-length lists.
    - Is a plausible (but clearly stylized and expressive) Indian English rendering of the input.
    - Uses durations, f0, and energy patterns that stay within a plausible human range and broadly similar numeric scale to the examples, but may be noticeably more varied, sing-song, and expressive.

    # Output Verbosity

    - Respond with exactly one valid Python tuple matching the required format.
    - Do not include any explanation, comments, or extra text.
    - Keep the tuple compact enough to fit comfortably on a single screen (e.g., ~15 lines or ≤300 tokens if needed), but always prioritize correctness and completeness of the tuple.

    # Examples and Supervision

    - Below are several real American→Indian example pairs from recordings.
    - Treat them as demonstrations of the desired transformation behavior and imitate their style for both sequences and numeric values, while allowing somewhat stronger pitch and energy variation than in the examples.
    """

    
    for i, (usa, indian) in enumerate(zip(icl_usa_data, icl_indian_data)):
        prompt += f"Example {i+1}:\n"
        prompt += f"American: {format_prosody_for_prompt(usa)}\n"
        prompt += f"Indian: {format_prosody_for_prompt(indian)}\n\n"
        
    prompt += "### Task:\n"
    prompt += f"Input American: {format_prosody_for_prompt(target_input)}\n"
    prompt += "Output Indian:"
    
    return prompt

def validate_output(output_str: str, input_len: int) -> Tuple[List[str], List[int], List[float], List[float]]:
    """
    Validates the LLM output.
    Checks if it's a tuple of 4 lists, if all lists have the same length,
    and if all phonemes (except special symbols) have duration > 0.
    """
    try:
        # cleanup potential markdown code blocks
        cleaned_output = output_str.strip()
        if cleaned_output.startswith("```python"):
            cleaned_output = cleaned_output[9:]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
        cleaned_output = cleaned_output.strip()

        data = ast.literal_eval(cleaned_output)
        
        if not isinstance(data, tuple) or len(data) != 4:
            raise ValueError("Output is not a tuple of 4 elements")
            
        phonemes, durations, pitch, energy = data
        
        if not (isinstance(phonemes, list) and isinstance(durations, list) and isinstance(pitch, list) and isinstance(energy, list)):
             raise ValueError("Output elements are not all lists")

        lengths = [len(phonemes), len(durations), len(pitch), len(energy)]
        if len(set(lengths)) != 1:
            raise ValueError(f"Lists have different lengths: ph={len(phonemes)}, dur={len(durations)}, pitch={len(pitch)}, energy={len(energy)}")
            
        # Check for zero durations on real phonemes
        special_symbols = {'_', '~', ' '}
        for i, (ph, dur) in enumerate(zip(phonemes, durations)):
            if ph not in special_symbols and dur <= 0:
                raise ValueError(f"Phoneme '{ph}' at index {i} has invalid duration {dur}. Real phonemes must have duration > 0.")
            
        return data
    except Exception as e:
        raise ValueError(f"Validation failed: {e}")

def call_llm(messages: List[Dict[str, str]], model: str, api_key: str) -> str:
    """Calls the OpenAI API with the provided messages."""
    client = openai.OpenAI(api_key=api_key)
    
    # No retry loop here; retries are handled in the main loop with updated messages
    try:
        logger.info(f"Sending request to OpenAI (Model: {model})")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0, # Deterministic output as requested
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert prosody using LLM")
    parser.add_argument("--input_file", required=True, help="Path to input USA prosody file")
    parser.add_argument("--output_file", required=True, help="Path to output Indian prosody file")
    parser.add_argument("--icl_usa", required=True, help="Path to USA ICL examples file")
    parser.add_argument("--icl_indian", required=True, help="Path to Indian ICL examples file")
    parser.add_argument("--api_key_file", default="openai_api_key.txt", help="Path to file containing OpenAI API key")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use (e.g., gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of lines to process (for testing)")
    parser.add_argument("--save_prompt", action="store_true", help="Save the initial prompt to a file for debugging")
    
    args = parser.parse_args()
    
    api_key = read_api_key(args.api_key_file)
    
    # Load ICL examples
    logger.info("Loading ICL examples...")
    with open(args.icl_usa, 'r') as f:
        icl_usa_lines = [parse_prosody_line(line) for line in f if line.strip()]
    with open(args.icl_indian, 'r') as f:
        icl_indian_lines = [parse_prosody_line(line) for line in f if line.strip()]
        
    if len(icl_usa_lines) != len(icl_indian_lines):
        logger.error("Mismatch in number of ICL examples")
        sys.exit(1)
        
    # Load input data
    logger.info(f"Loading input file: {args.input_file}")
    with open(args.input_file, 'r') as f:
        input_lines = [line.strip() for line in f if line.strip()]
        
    if args.limit:
        input_lines = input_lines[:args.limit]
        
    processed_count = 0
    with open(args.output_file, 'w') as f_out:
        for i, line in enumerate(input_lines):
            logger.info(f"Processing line {i+1}/{len(input_lines)}")
            try:
                input_data = parse_prosody_line(line)
                initial_prompt = construct_prompt(input_data, icl_usa_lines, icl_indian_lines)
                
                # Save prompt if requested (only for the first line to avoid spamming or overwriting)
                if args.save_prompt and i == 0:
                    prompt_file = "prompt_debug.txt"
                    logger.info(f"Saving initial prompt to {prompt_file}")
                    with open(prompt_file, "w") as pf:
                        pf.write(initial_prompt)

                # Initialize conversation history for this specific line
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that converts prosody data."},
                    {"role": "user", "content": initial_prompt}
                ]
                
                # Critique Loop
                valid_result = None
                for val_attempt in range(5):
                    try:
                        response_text = call_llm(messages, args.model, api_key)
                        valid_result = validate_output(response_text, len(input_data[0]))
                        break # Success
                    except ValueError as ve:
                        logger.warning(f"Validation failed on attempt {val_attempt+1}: {ve}")
                        # Append the error to the conversation history for the next attempt
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": f"Error: {ve}. Please correct your output."})
                        time.sleep(1) # Brief pause before retry
                    except Exception as e:
                        logger.error(f"API Error on attempt {val_attempt+1}: {e}")
                        time.sleep(2 ** val_attempt) # Exponential backoff for API errors
                
                if valid_result:
                    f_out.write(str(valid_result) + "\n")
                    f_out.flush()
                    processed_count += 1
                else:
                    logger.error(f"Failed to generate valid output for line {i+1} after 5 attempts.")
                    
            except Exception as e:
                logger.error(f"Error processing line {i+1}: {e}")
                
    logger.info(f"Processing complete. {processed_count}/{len(input_lines)} lines converted.")

if __name__ == "__main__":
    main()
