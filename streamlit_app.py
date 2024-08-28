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
    tokens = [int(x.strip()) for x in global_tokens.split(",")]
    return (q_idx in tokens) | (kv_idx in tokens)


def random_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return random() < frac_random


def causal_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return kv_idx <= q_idx


def dilated_sliding_mask(
    batch_idx: int, head_idx: int, q_idx: int, kv_idx: int
) -> bool:
    diff = abs(q_idx - kv_idx)
    return (diff <= window_parameter) & (diff % dilation_factor == 0)


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
                "dilation_factor": dilation_factor,
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
seq_len = st.sidebar.slider("Sequence Length", 16, max_value=128, value=24)
window_parameter = st.sidebar.number_input(
    "Sliding Window size", min_value=0, max_value=seq_len, value=5, step=1
)
frac_random = st.sidebar.number_input(
    "Fraction of Random Mask", min_value=0.0, max_value=1.0, value=0.1, step=0.05
)
dilation_factor = st.sidebar.number_input(
    "Dilation Factor", min_value=1, max_value=5, value=2, step=1
)
global_tokens = st.sidebar.text_input(
    "Global Tokens as comma seperarted values", value="0"
)

# Add a selectbox to choose between masking functions
masking_options = {
    "No Mask": default_mask_mod,
    "Sliding Window Mask": sliding_window_mask,
    "Dilated Sliding Mask": dilated_sliding_mask,
    "Prefix Token Mask": prefix_token_mask,
    "Random Mask": random_mask,
    "BigBird Mask": bigbird_mask_mod,
    "Causal Mask": causal_mask,
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
st.markdown(
    "#### [Flex Attention](https://pytorch.org/blog/flexattention/) Mask Visualization"
)
st.markdown(
    """
    Mask for the selected attention mechanism.
    Modify the function with optional global parameters (`window_parameter`, `frac_random`, `dilation_factor`, `global_tokens`) to visualize output pattern.
    """
)
mask = flex_attention(mask_mod_func, 1, 1, seq_len)

# Visualize the mask for the selected batch and head
fig, ax = plt.subplots(figsize=(8, 8))

# Show the mask
cax = ax.pcolor(mask[0, 0], cmap="viridis", edgecolors="black", linewidths=2)

ax.set_xlabel("Key/Value Position")
ax.set_ylabel("Query Position")

ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

# Set the number of ticks to match sequence length
ax.set_xticks(np.arange(seq_len) + 0.5, minor=False)
ax.set_yticks(np.arange(seq_len) + 0.5, minor=False)

# Modify x-label and y-label by adding the prefix Q and KV
ax.set_xticklabels([f"KV{i + 1}" for i in range(seq_len)], rotation=45)
ax.set_yticklabels([f"Q{i + 1}" for i in range(seq_len)])

# Invert the y-axis to place the origin at the top-left
ax.invert_yaxis()

# Add color bar for reference
fig.colorbar(cax, ax=ax, use_gridspec=True)

st.pyplot(fig)
