import re
import json
from typing import Optional, Union, Dict, List

def extract_clean_json(
    response_text: str,
    remove_thinking_tags: bool = True
) -> Optional[Union[Dict, List]]:
    """
    Extracts the first valid JSON object found in a string, optionally after
    removing DeepSeek-style <thinking>...</thinking> blocks.

    Args:
        response_text: The raw string response which might contain thinking tags
                       and a JSON object.
        remove_thinking_tags: If True (default), removes <thinking>...</thinking>
                              blocks before searching for JSON.

    Returns:
        The parsed JSON object (as a dict or list) if found and valid,
        otherwise None.
    """
    if not isinstance(response_text, str):
        # Handle cases where input might not be a string
        try:
            response_text = str(response_text)
        except Exception:
            return None # Cannot proceed if input is not stringifiable

    text_to_parse = response_text

    # 1. Optionally remove <thinking> tags
    if remove_thinking_tags:
        # Regex to find <thinking> ... </thinking> blocks, including multi-line content
        # DOTALL flag makes '.' match newlines as well
        # Non-greedy '.*?' ensures it stops at the *first* closing tag
        thinking_pattern = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
        text_to_parse = thinking_pattern.sub("", text_to_parse).strip()
        # print(f"--- Text after removing thinking tags ---\n{text_to_parse}\n---") # Debugging line

    if not text_to_parse:
        return None

    # 2. Find the start and end of the first potential JSON object
    # Look for the first '{' or '['
    first_brace = text_to_parse.find('{')
    first_bracket = text_to_parse.find('[')

    if first_brace == -1 and first_bracket == -1:
        # No JSON object indicator found
        # print("--- No JSON start indicator found ---") # Debugging line
        return None

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_index = first_brace
        end_char = '}'
    else:
        start_index = first_bracket
        end_char = ']'

    # Look for the last matching '}' or ']'
    # This is a heuristic: assumes the JSON is the main/last block
    # More robust parsing might be needed for complex cases with multiple JSONs
    end_index = -1
    nesting_level = 0
    potential_end = -1

    # Iterate from the start index to find the corresponding closing bracket/brace
    # This is more reliable than just rfind for nested structures
    balance_check_required = True
    if end_char == '}':
        open_char = '{'
    else: # end_char == ']'
        open_char = '['

    # Simple rfind first as a quick check (often works for simple LLM outputs)
    potential_end_quick = text_to_parse.rfind(end_char)
    if potential_end_quick > start_index:
        json_candidate_quick = text_to_parse[start_index : potential_end_quick + 1]
        try:
            parsed_json = json.loads(json_candidate_quick)
            # print(f"--- Successfully parsed using simple rfind ---") # Debugging line
            return parsed_json
        except json.JSONDecodeError:
             # print(f"--- Simple rfind failed, proceeding to balance check ---") # Debugging line
             pass # Fall through to balance check


    # If simple rfind didn't work, do a proper balance check
    # print(f"--- Starting balance check from index {start_index} ---") # Debugging line
    for i in range(start_index, len(text_to_parse)):
        char = text_to_parse[i]
        if char == open_char:
            nesting_level += 1
        elif char == end_char:
            nesting_level -= 1
            if nesting_level == 0:
                # Found the matching end character for the initial start
                potential_end = i
                break # Stop searching once the initial scope is closed

    if potential_end == -1:
        # print(f"--- Could not find matching end character using balance check ---") # Debugging line
        return None # No matching end found

    # 3. Extract the potential JSON string
    json_candidate = text_to_parse[start_index : potential_end + 1]
    # print(f"--- JSON Candidate (Balance Check) ---\n{json_candidate}\n---") # Debugging line

    # 4. Try to parse the extracted string
    try:
        parsed_json = json.loads(json_candidate)
        return parsed_json
    except json.JSONDecodeError as e:
        # print(f"--- JSONDecodeError: {e} ---") # Debugging line
        # Optional: Add more fallback logic here if needed, e.g., trying to
        # find JSON within code blocks (```json ... ```)
        return None # Failed to parse the candidate string
