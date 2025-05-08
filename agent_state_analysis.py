import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import io

# Title and description
st.set_page_config(page_title="Agent State Analysis", layout="wide")
st.title("Agent State Analysis & Comparison")
st.markdown(
    """
    Upload agent state `.npz` files (and optional world state) to interactively examine
    individual state variables, visualize memory patterns, and compare across agents.
    """
)

# Sidebar for file uploads
st.sidebar.header("Upload State Files")
agent1_file = st.sidebar.file_uploader("Agent 1 State (.npz)", type=["npz"], key="a1")
agent2_file = st.sidebar.file_uploader("Agent 2 State (.npz)", type=["npz"], key="a2")
world_file  = st.sidebar.file_uploader("World State (.npz, optional)", type=["npz"], key="w")

# Function to load .npz files
def load_npz(source, label):
    try:
        return np.load(source, allow_pickle=True)
    except Exception as e:
        st.sidebar.error(f"Failed to load {label} from {source}: {e}")
        return None

# Load uploaded states or defaults
state1 = load_npz(agent1_file, "Agent 1") if agent1_file else None
state2 = load_npz(agent2_file, "Agent 2") if agent2_file else None
world  = load_npz(world_file,  "World")    if world_file  else None

# Helper to try defaults
def try_default(path, current, label):
    if current is None and os.path.exists(path):
        default = load_npz(path, label)
        if default is not None:
            st.sidebar.info(f"Loaded default {label} state from {path}")
            return default
    return current

state1 = try_default("agent_1_state.npz", state1, "Agent 1")
state2 = try_default("agent_2_state.npz", state2, "Agent 2")
world  = try_default("world_state.npz",   world,  "World")

# Summary display
def summarize_state(state, label):
    st.subheader(f"Summary: {label}")
    if state is None:
        st.info("No file uploaded or default available.")
        return
    summary = []
    for k in state.files:
        try:
            arr = state[k]
            shape_str = str(arr.shape) if hasattr(arr, 'shape') else str(type(arr))
            if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number) and arr.size > 0:
                mean_val = float(np.round(np.mean(arr), 4))
                std_val = float(np.round(np.std(arr), 4))
            else:
                mean_val = None
                std_val = None
            summary.append({"Key": k, "Shape": shape_str, "Mean": mean_val, "Std": std_val})
        except Exception as e:
            summary.append({"Key": k, "Shape": "<unreadable>", "Mean": None, "Std": None})
            st.warning(f"Could not summarize '{k}': {e}")
    st.table(summary)

# Layout: summaries side by side
col1, col2 = st.columns(2)
with col1:
    summarize_state(state1, "Agent 1")
with col2:
    summarize_state(state2, "Agent 2")

# Visualization section
st.markdown("---")
st.header("Visualize State Variables")
all_states = {"Agent 1": state1, "Agent 2": state2, "World": world}
select_label = st.selectbox("Select State to Inspect", [k for k, v in all_states.items() if v])
selected_state = all_states[select_label]
if selected_state:
    key = st.selectbox("Select Variable", selected_state.files)
    arr = selected_state[key]
    st.write(f"**Shape:** {getattr(arr, 'shape', 'n/a')}" )
    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
        # Create plot
        fig, ax = plt.subplots()
        if arr.ndim == 1:
            ax.hist(arr, bins=50)
            ax.set_title(f"Histogram of {key}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        elif arr.ndim == 2:
            im = ax.imshow(arr[:100, :], aspect='auto', cmap='viridis')
            ax.set_title(f"Heatmap of {key} (first 100 rows)")
            fig.colorbar(im, ax=ax)
        else:
            st.write(arr)
            fig = None
        # Render smaller using st.image
        if fig:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            st.image(buf, width=400)
            plt.close(fig)
    else:
        st.write(arr)

# Comparison section
if state1 and state2:
    st.markdown("---")
    st.header("Compare Agent 1 vs Agent 2")
    common_keys = sorted(set(state1.files) & set(state2.files))
    if not common_keys:
        st.warning("No common variables to compare.")
    else:
        key = st.selectbox("Select Variable to Compare", common_keys, key="compare")
        arr1, arr2 = state1[key], state2[key]
        if (isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)
                and np.issubdtype(arr1.dtype, np.number) and arr1.shape == arr2.shape):
            diff = arr1 - arr2
            st.write(f"**Shape:** {arr1.shape}")
            st.metric("Mean Difference", float(np.round(np.mean(diff), 4)))
            st.metric("Max Abs Difference", float(np.round(np.max(np.abs(diff)), 4)))
            fig, ax = plt.subplots()
            ax.hist(diff.flatten(), bins=50)
            ax.set_title(f"Difference Histogram: {key}")
            ax.set_xlabel("Difference")
            ax.set_ylabel("Count")
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            st.image(buf, width=400)
            plt.close(fig)
        elif (isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)
              and arr1.shape == arr2.shape):
            flat1, flat2 = arr1.flatten(), arr2.flatten()
            total = flat1.size
            mismatches = sum(1 for a, b in zip(flat1, flat2) if a != b)
            st.write(f"**Shape:** {arr1.shape}")
            st.metric("Total Elements", total)
            st.metric("Mismatches", mismatches)
            st.metric("Mismatch Rate", f"{round(mismatches/total*100, 2)}%")
            if mismatches > 0:
                samples = []
                for idx, (a, b) in enumerate(zip(flat1, flat2)):
                    if a != b:
                        samples.append({"Index": idx, "Agent1": str(a), "Agent2": str(b)})
                    if len(samples) >= 10:
                        break
                st.subheader("Sample Mismatches")
                st.table(samples)
        else:
            st.error(f"Cannot compare '{key}' (non-matching shapes or unsupported types)")

st.sidebar.markdown("---")
