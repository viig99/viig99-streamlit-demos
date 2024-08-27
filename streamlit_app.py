import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import ast
from random import random
from streamlit_ace import st_ace
import inspect


def default_mask_mod(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return True


def sliding_window_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return abs(q_idx - kv_idx) <= window_parameter


def prefix_token_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return q_idx == 0 or kv_idx == 0


def random_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return random() < frac_random


def bigbird_mask_mod(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    # Sliding window mask
    slw_mask = abs(q_idx - kv_idx) <= window_parameter
    # Global attention mask
    tokens = [int(x.strip()) for x in global_tokens.split(",")]
    prefix_lm = q_idx in tokens or kv_idx in tokens
    # Random mask
    r_mask = random() < frac_random

    return slw_mask or prefix_lm or r_mask


def flex_attention(mask_mod_func, batch_size, num_heads, sequence_length):
    new_mask = np.ones((batch_size, num_heads, sequence_length, sequence_length))
    for b in range(batch_size):
        for h in range(num_heads):
            for q_idx in range(sequence_length):
                for kv_idx in range(sequence_length):
                    new_mask[b, h, q_idx, kv_idx] = mask_mod_func(b, h, q_idx, kv_idx)
    return new_mask


def check_signature_matches(default_func, user_func):
    # Get signatures of the default and user-defined functions
    default_sig = inspect.signature(default_func)
    user_sig = inspect.signature(user_func)

    # Compare signatures
    if default_sig != user_sig:
        st.error(
            f"Error: Function signature mismatch. Expected {default_sig}, but got {user_sig}."
        )
        return False
    return True


def parse_function_from_string(source_code: str):
    try:
        # Parse the source code into an AST
        parsed_code = ast.parse(source_code, filename="<ast>")

        # Check that the root is a module and has exactly one function definition
        if len(parsed_code.body) != 1 or not isinstance(
            parsed_code.body[0], ast.FunctionDef
        ):
            raise ValueError("Only one function definition is allowed.")

        # Compile the AST into a code object
        code = compile(parsed_code, filename="<ast>", mode="exec")

        # Create a namespace for the function
        local_context = {}
        exec(
            code,
            {
                "window_parameter": window_parameter,
                "frac_random": frac_random,
                "global_tokens": global_tokens,
                "random": random,
            },
            local_context,
        )

        # Return the parsed function
        user_func = local_context.get(parsed_code.body[0].name, None)

        # Check if the user function has the same signature as the default function
        if check_signature_matches(default_mask_mod, user_func):
            return user_func

    except Exception as e:
        st.error(f"Error parsing function: {e}")
        return None


# Set up the Streamlit sidebar for user inputs
st.sidebar.header("Global Configurable Parameters")
seq_len = st.sidebar.slider("Sequence Length", 32, max_value=128, value=64)
window_parameter = st.sidebar.number_input(
    "Sliding Window size", min_value=0, max_value=seq_len, value=5, step=1
)
frac_random = st.sidebar.number_input(
    "Fraction of Random Mask", min_value=0.0, max_value=1.0, value=0.1, step=0.05
)
global_tokens = st.sidebar.text_input(
    "Global Tokens as comma seperarted values", value="0"
)

# Add a selectbox to choose between masking functions
masking_options = {
    "No Mask": default_mask_mod,
    "Sliding Window Mask": sliding_window_mask,
    "Prefix Token Mask": prefix_token_mask,
    "Random Mask": random_mask,
    "BigBird Mask": bigbird_mask_mod,
}

option = st.sidebar.selectbox(
    "Select Masking Function",
    list(masking_options.keys()),
    index=list(masking_options.keys()).index("BigBird Mask"),
)

# Set the ACE editor with the selected function's code
content = st_ace(
    language="python", value=inspect.getsource(masking_options[option]), height=200
)

# Parse the user-defined function
mask_mod_func = parse_function_from_string(content)

# Fallback to selected option if parsing failed
if mask_mod_func is None:
    mask_mod_func = masking_options[option]

# Generate the mask based on user input
st.markdown("#### Flex Attention Mask Visualization")
st.markdown(
    """
    Mask for the selected attention mechanism.
    Modify the function with optional global parameters (`window_parameter`, `frac_random`, `global_tokens`) to visualize output pattern.
    """
)
mask = flex_attention(mask_mod_func, 1, 1, seq_len)

# Visualize the mask for the selected batch and head
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(mask[0, 0], cmap="viridis")
ax.set_xlabel("Sequence Position")
ax.set_ylabel("Sequence Position")
st.pyplot(fig)
